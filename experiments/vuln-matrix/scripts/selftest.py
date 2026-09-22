#!/usr/bin/env python3
"""Torch-free selftest of the parts that carry the experiment's logic.

Runs anywhere Python and the stdlib are available -- no torch, no torchvision,
no dataset. Covers percentile pools, deterministic mixing, sentinel injection,
stage-resume equivalence, warm-up percentile arithmetic, and the
misclassification / wasted-CPU maths in the summariser. It does NOT cover the
torch pipelines; scripts/smoke_test.sh does that.
"""

import csv
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmp import manifest as M                      # noqa: E402
from hmp.minato import (                           # noqa: E402
    EVENT_FIELDS, StageAborted, calibrate_t_out, run_stages,
)
from hmp.adaptive import (                        # noqa: E402
    DEFAULTS as ADAPT_DEFAULTS, SharedThreshold, ThresholdController,
)
from hmp.runtime import summarise                  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    tag = "  PASS  " if cond else "  FAIL  "
    print(tag + name + (("  (%s)" % detail) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def fake_profile(n=400):
    """Latencies spanning two orders of magnitude, heavy-tailed."""
    rows = []
    for i in range(n):
        if i % 50 == 0:
            lat = 1.0 + i * 0.01          # extreme tail
        elif i % 4 == 0:
            lat = 0.30 + (i % 17) * 0.01  # moderate tail
        else:
            lat = 0.05 + (i % 11) * 0.002
        rows.append({"sample_id": "s%04d" % i, "path": "/tmp/s%04d" % i,
                     "bytes": 1000 + i, "latency_s": lat})
    return rows


def test_pools():
    print("\n--- pools and ratio regimes ---")
    rows = fake_profile()
    f_mod, s_mod, cuts_mod = M.build_pools(rows, "moderate")
    f_ext, s_ext, cuts_ext = M.build_pools(rows, "extreme")

    check("fast pool is the same in both regimes (P0-P50)",
          set(r["sample_id"] for r in f_mod) == set(r["sample_id"] for r in f_ext),
          "%d rows" % len(f_mod))
    check("moderate and extreme slow pools are disjoint",
          not (set(r["sample_id"] for r in s_mod) &
               set(r["sample_id"] for r in s_ext)),
          "moderate=%d extreme=%d" % (len(s_mod), len(s_ext)))
    check("every fast-pool latency is below every slow-pool latency",
          max(r["latency_s"] for r in f_mod) <= min(r["latency_s"] for r in s_mod))
    check("extreme regime has a larger cost ratio than moderate",
          cuts_ext["observed_cost_ratio"] > cuts_mod["observed_cost_ratio"],
          "%.1fx vs %.1fx" % (cuts_ext["observed_cost_ratio"],
                              cuts_mod["observed_cost_ratio"]))
    check("fast pool is about half the candidates",
          0.4 * len(rows) <= len(f_mod) <= 0.6 * len(rows),
          "%d of %d" % (len(f_mod), len(rows)))


def test_mixing():
    print("\n--- deterministic mixing and sentinels ---")
    rows = fake_profile()
    fast, slow, _ = M.build_pools(rows, "moderate")
    sent = M.choose_sentinels(fast, slow)

    for frac in (0.10, 0.25, 0.40, 0.60):
        mixed, meta = M.mix_dataset(fast, slow, 200, frac, seed=7, sentinels=sent)
        n_slow = sum(1 for r in mixed if r["truth_label"] == "slow")
        ok = len(mixed) == 200 and n_slow == int(round(200 * frac))
        check("slow_fraction %.2f produces the exact requested proportion" % frac,
              ok, "%d/%d slow" % (n_slow, len(mixed)))

    a, _ = M.mix_dataset(fast, slow, 200, 0.25, seed=7, sentinels=sent)
    b, _ = M.mix_dataset(fast, slow, 200, 0.25, seed=7, sentinels=sent)
    c, _ = M.mix_dataset(fast, slow, 200, 0.25, seed=8, sentinels=sent)
    check("same seed reproduces the manifest exactly",
          M.manifest_fingerprint(a) == M.manifest_fingerprint(b))
    check("different seed produces a different manifest",
          M.manifest_fingerprint(a) != M.manifest_fingerprint(c))

    present = []
    for frac in (0.10, 0.25, 0.40, 0.60):
        mixed, meta = M.mix_dataset(fast, slow, 200, frac, seed=7, sentinels=sent)
        ids = set(r["sample_id"] for r in mixed)
        present.append(sent["fast"]["sample_id"] in ids and
                       sent["slow"]["sample_id"] in ids)
    check("both sentinels appear at every slow_fraction", all(present))

    mixed, _ = M.mix_dataset(fast, slow, 200, 0.25, seed=7, sentinels=sent)
    roles = sorted(r["sentinel_role"] for r in mixed if r["is_sentinel"] == "1")
    check("sentinels are labelled with their role", roles == ["fast", "slow"],
          str(roles))

    labelled = all(r["truth_label"] in ("fast", "slow") for r in mixed)
    check("every row carries a ground-truth pool label", labelled)


def test_resume():
    print("\n--- stage-resume equivalence ---")

    def mk(k):
        def stage(payload):
            payload["v"] = [x * 2 + k for x in payload["v"]]
            return payload
        return stage

    stages = [("s%d" % k, mk(k)) for k in range(6)]
    base = {"row": {"sample_id": "x"}, "v": [1.0, 2.0, 3.0]}

    full = run_stages(stages, dict(base, v=list(base["v"])))
    ok = True
    for k in range(len(stages) + 1):
        sink = {}
        mid = run_stages(stages[:k], dict(base, v=list(base["v"])),
                         checkpoint=lambda s, d: sink.setdefault(s, d))
        resumed = run_stages(stages, mid, start_index=k)
        ok = ok and resumed["v"] == full["v"]
    check("resuming at every stage index reproduces the full pipeline", ok)

    class InPlace(object):
        def __call__(self, payload):
            payload["v"].append(99.0)
            return payload

    seen = {}
    run_stages([("a", mk(1)), ("b", InPlace()), ("c", InPlace())],
               dict(base, v=list(base["v"])),
               checkpoint=lambda s, d: seen.setdefault(s, d))
    check("a published checkpoint is not mutated by later in-place stages",
          len(seen[1]["v"]) == 3, "len=%d" % len(seen[1]["v"]))

    import threading
    abort = threading.Event()
    timings = []
    pub = {}

    def cb(stage, payload):
        pub["stage"] = stage
        if stage == 2:
            abort.set()

    aborted = False
    try:
        run_stages(stages, dict(base, v=list(base["v"])), checkpoint=cb,
                   abort_event=abort, timings=timings)
    except StageAborted:
        aborted = True
    check("abort stops at a stage boundary with one stage lost",
          aborted and len(timings) == 2 and pub["stage"] == 2,
          "%d stages ran" % len(timings))


def test_calibration():
    print("\n--- warm-up calibration ---")
    import time as _t

    latencies = [0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                 0.11, 0.12, 0.14, 0.18, 0.22, 0.60, 0.80, 1.00, 1.40, 2.00]
    rows = [{"sample_id": "c%02d" % i, "path": "",
             "truth_label": "slow" if latencies[i] > 0.3 else "fast"}
            for i in range(len(latencies))]

    def stage(payload):
        idx = int(payload["row"]["sample_id"][1:])
        deadline = _t.perf_counter() + latencies[idx] * 0.05
        while _t.perf_counter() < deadline:
            pass
        return payload

    stages = [("only", stage)]
    res = calibrate_t_out(stages, rows, warmup_samples=len(rows),
                          percentile=75.0, seed=3, discard=0,
                          log=lambda *a: None)

    check("warm-up timed every sample", res["n_warmup"] == len(rows),
          "%d" % res["n_warmup"])
    check("t_out equals P75 of the measured distribution",
          abs(res["t_out"] - res["latency"]["p75"]) < 1e-12)
    check("t_out is not a hardcoded constant",
          res["latency"]["p50"] < res["t_out"] < res["latency"]["p100"])
    marg = calibrate_t_out(stages, rows, len(rows), 75.0, 3, margin=2.0,
                           discard=0, log=lambda *a: None)
    check("margin multiplies the percentile",
          abs(marg["t_out"] - 2.0 * marg["t_out_raw"]) < 1e-12,
          "t_out=%.6f raw=%.6f" % (marg["t_out"], marg["t_out_raw"]))
    check("per-label warm-up breakdown is recorded",
          set(res["warmup_by_truth_label"]) == {"fast", "slow"})
    check("fraction of the slow pool falling below t_out is reported",
          "warmup_slow_below_t_out_frac" in res,
          "%.2f at slow_fraction=0.25" % res["warmup_slow_below_t_out_frac"])

    # Vulnerability #2, in miniature: once the slow pool is more than a quarter
    # of the mix, the P75 cut sits inside it and part of the slow population is
    # classified fast by construction.
    heavy = [dict(r) for r in rows]
    for i in range(6, len(heavy)):          # 70% of the mix is now slow
        heavy[i]["truth_label"] = "slow"
    res_heavy = calibrate_t_out(stages, heavy, len(heavy), 75.0, 3, discard=0,
                                log=lambda *a: None)
    check("above 25% slow, part of the slow pool falls below the P75 cut",
          res_heavy["warmup_slow_below_t_out_frac"] > 0.5,
          "%.2f of slow samples below t_out"
          % res_heavy["warmup_slow_below_t_out_frac"])


def _controller(t0=0.20, **over):
    cfg = dict(ADAPT_DEFAULTS)
    cfg.update({"adapt": True, "adapt_min_samples": 4, "adapt_cooldown_s": 0.0,
                "adapt_interval_s": 0.0, "adapt_window": 256,
                "adapt_max_step_ratio": 0.0, "t_out_min": 0.001})
    cfg.update(over)
    shared = SharedThreshold(t0, cfg["adapt_percentile"])
    return ThresholdController(shared, None, cfg), shared


def test_adaptation():
    print("\n--- dynamic threshold adaptation ---")

    # 1. the recomputed threshold is the P75 of what the workers reported
    ctrl, shared = _controller()
    obs = [0.05, 0.06, 0.07, 0.09, 0.10, 0.12, 0.15, 0.20, 0.40, 0.90]
    for i, v in enumerate(obs):
        ctrl._ingest(("fast", i, v))
    ctrl._maybe_update(force=True)
    import numpy as _np
    check("t_out is recomputed as P75 of the live observations",
          abs(shared.get() - float(_np.percentile(obs, 75))) < 1e-9,
          "%.4f" % shared.get())
    check("an update is traced", ctrl.updates == 1)

    # 2. a high migration rate escalates P75 -> P90
    ctrl, shared = _controller(adapt_escalate_rate=0.35)
    for i in range(10):
        ctrl._ingest(("fast", i, 0.05))
    for i in range(10, 20):                       # 50% migration rate
        ctrl._ingest(("migrate", i, 0.20))
        ctrl._ingest(("resume", i, 0.30))
    ctrl._maybe_update(force=True)
    check("migration rate above the escalate threshold falls back to P90",
          abs(shared.get_percentile() - 90.0) < 1e-9 and ctrl.escalations == 1,
          "percentile=%.0f" % shared.get_percentile())
    check("the P90 threshold is above what P75 would have given",
          shared.get() > float(_np.percentile(sorted(ctrl.window), 75)) - 1e-12,
          "t_out=%.4f" % shared.get())

    # 3. once the rate falls, it returns to P75
    for i in range(200, 400):
        ctrl._ingest(("fast", i, 0.05))
    ctrl._maybe_update(force=True)
    check("a low migration rate de-escalates back to P75",
          abs(shared.get_percentile() - 75.0) < 1e-9 and ctrl.deescalations == 1,
          "percentile=%.0f" % shared.get_percentile())

    # 4. reconstructed mode pairs a migration with its resume
    ctrl, _ = _controller(adapt_censoring="reconstructed")
    ctrl._ingest(("migrate", 1, 0.20))
    check("a migration alone contributes no observation yet",
          len(ctrl.window) == 0)
    ctrl._ingest(("resume", 1, 0.35))
    check("the observation is detect_s + bg_s once the resume arrives",
          len(ctrl.window) == 1 and abs(ctrl.window[0] - 0.55) < 1e-9,
          "%.4f" % ctrl.window[0])
    ctrl._ingest(("resume", 999, 0.1))
    check("an unmatched resume is ignored", len(ctrl.window) == 1)

    # 5. censored feedback pins t_out; it cannot track a growing tail.
    #    40% of samples truly cost 1.0s, so the correct P75 is 1.0s.
    def drive(mode, t0, rounds=12, slow_frac=0.40, slow_cost=1.0, n=40):
        ctrl, shared = _controller(t0=t0, adapt_censoring=mode,
                                   adapt_escalate_rate=1.01)  # no escalation
        traj = [t0]
        for _ in range(rounds):
            t = shared.get()
            for i in range(n):
                if i < n * (1 - slow_frac):
                    ctrl._ingest(("fast", i, 0.05))
                else:
                    spent = min(t, slow_cost)
                    ctrl._ingest(("migrate", i, spent))
                    ctrl._ingest(("resume", i, max(0.0, slow_cost - spent)))
            ctrl._maybe_update(force=True)
            traj.append(shared.get())
        return traj

    cens = drive("censored", 0.20)
    recon = drive("reconstructed", 0.20)
    check("censored feedback pins t_out at its starting value",
          max(cens) - min(cens) < 1e-9, "stayed at %.4f for %d updates"
          % (cens[-1], len(cens) - 1))
    check("censored feedback cannot reach the true P75",
          cens[-1] < 0.5, "t_out=%.4f, true P75=1.0" % cens[-1])
    check("reconstructed feedback converges on the true P75",
          abs(recon[-1] - 1.0) < 1e-6, "t_out=%.4f" % recon[-1])
    check("censored feedback is non-increasing under load",
          all(b <= a + 1e-12 for a, b in zip(cens, cens[1:])))

    # 6. the step limiter damps a single large move
    ctrl, shared = _controller(t0=0.10, adapt_max_step_ratio=1.5)
    for i in range(20):
        ctrl._ingest(("fast", i, 5.0))
    ctrl._maybe_update(force=True)
    check("a single update cannot move t_out by more than the step ratio",
          abs(shared.get() - 0.15) < 1e-9, "t_out=%.4f" % shared.get())

    # 7. the shared value is what a worker would read
    ctrl, shared = _controller(t0=0.33)
    from hmp.adaptive import WorkerThresholdView
    view = WorkerThresholdView(shared.handles())
    check("a worker view observes the shared threshold",
          abs(view.get() - 0.33) < 1e-12)
    shared.set(0.77, 90.0)
    check("a worker view observes a rewritten threshold",
          abs(view.get() - 0.77) < 1e-12 and abs(view.get_percentile() - 90) < 1e-12)


def _write_events(run_dir, records):
    d = os.path.join(run_dir, "events")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "events_fg_1.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EVENT_FIELDS)
        w.writeheader()
        for r in records:
            row = {k: "" for k in EVENT_FIELDS}
            row.update(r)
            w.writerow(row)


def test_metrics():
    print("\n--- misclassification and wasted-CPU maths ---")
    tmp = tempfile.mkdtemp(prefix="hmp_selftest_")
    try:
        run_dir = os.path.join(tmp, "run")
        os.makedirs(run_dir, exist_ok=True)
        recs = []
        # 6 slow samples: 4 correctly migrated, 2 kept fast (false negatives)
        for i in range(4):
            recs.append({"event": "migrate", "sample_id": "slow%d" % i,
                         "truth_label": "slow", "duration_s": "0.20",
                         "detect_s": "0.20", "preserved_s": "0.05",
                         "wasted_s": "0.15", "resume_stage": "2", "n_stages": "6"})
            recs.append({"event": "resume", "sample_id": "slow%d" % i,
                         "truth_label": "slow", "duration_s": "0.30",
                         "resume_stage": "2", "n_stages": "6"})
        for i in range(4, 6):
            recs.append({"event": "fast", "sample_id": "slow%d" % i,
                         "truth_label": "slow", "duration_s": "0.19",
                         "resume_stage": "6", "n_stages": "6"})
        # 10 fast samples: 1 wrongly migrated (false positive)
        recs.append({"event": "migrate", "sample_id": "fast0",
                     "truth_label": "fast", "duration_s": "0.20",
                     "detect_s": "0.20", "preserved_s": "0.10",
                     "wasted_s": "0.10", "resume_stage": "0", "n_stages": "6"})
        recs.append({"event": "resume_restart", "sample_id": "fast0",
                     "truth_label": "fast", "duration_s": "0.25",
                     "resume_stage": "0", "n_stages": "6"})
        for i in range(1, 10):
            recs.append({"event": "fast", "sample_id": "fast%d" % i,
                         "truth_label": "fast", "duration_s": "0.05",
                         "resume_stage": "6", "n_stages": "6"})
        _write_events(run_dir, recs)

        meta = {"n_samples": 16, "stage_median_s": {"a": 0.1, "b": 0.2}}
        row = summarise(run_dir, meta)

        check("true positives counted", row["tp_slow_migrated"] == 4,
              str(row["tp_slow_migrated"]))
        check("false negatives counted (slow kept in the fast queue)",
              row["fn_slow_kept_fast"] == 2, str(row["fn_slow_kept_fast"]))
        check("false positives counted", row["fp_fast_migrated"] == 1,
              str(row["fp_fast_migrated"]))
        check("true negatives counted", row["tn_fast_kept_fast"] == 9,
              str(row["tn_fast_kept_fast"]))
        check("misclassification rate = (fp+fn)/n",
              abs(row["misclassification_rate"] - 3 / 16.0) < 1e-9,
              "%.4f" % row["misclassification_rate"])
        check("slow recall = tp/(tp+fn)",
              abs(row["slow_recall"] - 4 / 6.0) < 1e-9,
              "%.4f" % row["slow_recall"])
        check("detection tax is the sum of pre-classification time",
              abs(row["detect_tax_total_s"] - 1.0) < 1e-9,
              "%.3fs" % row["detect_tax_total_s"])
        check("detection tax per sample divides by the dataset size",
              abs(row["detect_tax_per_sample_s"] - 1.0 / 16) < 1e-9)
        check("stage-0 migrations counted separately",
              row["migrations_at_stage_0"] == 1,
              str(row["migrations_at_stage_0"]))
        check("resume-from-index waste is below full-restart waste",
              row["wasted_cpu_resume_s"] < row["wasted_cpu_full_restart_s"],
              "%.3fs vs %.3fs" % (row["wasted_cpu_resume_s"],
                                  row["wasted_cpu_full_restart_s"]))
        check("sentinel columns exist even with no sentinels present",
              "sentinel_fast_p50_s" in row)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("H-AIP vulnerability-matrix selftest (torch-free)")
    test_pools()
    test_mixing()
    test_resume()
    test_calibration()
    test_adaptation()
    test_metrics()
    print("\n" + "=" * 60)
    if FAILURES:
        print("FAILED %d check(s): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
