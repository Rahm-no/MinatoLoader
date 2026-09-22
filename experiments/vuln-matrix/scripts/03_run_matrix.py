#!/usr/bin/env python3
"""Script 3 -- run PyTorch DataLoader and MinatoLoader F/B splits over one
extracted dataset manifest, one full epoch each.

    python3 scripts/03_run_matrix.py \
        --manifest manifests/openimages_moderate_f0.25_n20000.csv \
        --pipeline openimages \
        --splits 11F1B,9F3B,7F5B,5F7B,3F9B \
        --out-root results/openimages

Writes one directory per run plus an appended row in <out-root>/summary.csv.
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmp import manifest as M            # noqa: E402
from hmp import pipelines                # noqa: E402
from hmp.adaptive import DEFAULTS as ADAPT_DEFAULTS   # noqa: E402
from hmp.minato import MinatoHarness, calibrate_t_out   # noqa: E402
from hmp.runtime import (                # noqa: E402
    Consumer, SystemMonitor, append_summary, baseline_loader, plan_cores,
    summarise,
)

BATCH_FIELDS = ["batch_index", "size", "latency_s", "mean_item_wait_s",
                "n_truth_slow", "n_route_slow", "n_error", "consumer_s"]


def parse_split(text):
    """'9F3B' -> (9, 3)"""
    t = text.strip().upper()
    f, _, rest = t.partition("F")
    b = rest.replace("B", "")
    return int(f), int(b)


def write_batches(path, records):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BATCH_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in BATCH_FIELDS})


def base_cfg(args, rows, run_dir):
    return {
        "pipeline": args.pipeline,
        "pipeline_kwargs": json.loads(args.pipeline_kwargs),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "events_dir": os.path.join(run_dir, "events"),
        "front_threads": args.front_threads,
        "baseline_workers": args.baseline_workers,
        "fast_queue_size": args.fast_queue_size,
        "slow_drain_threshold": args.slow_drain_threshold,
        "queue_poll_s": args.queue_poll_s,
        "queue_trace_interval_s": args.queue_trace_interval,
        "idle_timeout_s": args.idle_timeout,
        "epoch_timeout_s": args.epoch_timeout,
        "pin_memory": args.pin_memory,
        "max_batches": args.max_batches,
        # dynamic threshold adaptation (paper section 4.2)
        "adapt": args.adapt,
        "adapt_percentile": args.t_out_percentile,
        "adapt_fallback_percentile": args.adapt_fallback_percentile,
        "adapt_margin": args.t_out_margin,
        "adapt_window": args.adapt_window,
        "adapt_interval_s": args.adapt_interval,
        "adapt_min_samples": args.adapt_min_samples,
        "adapt_max_step_ratio": args.adapt_max_step_ratio,
        "adapt_escalate_rate": args.adapt_escalate_rate,
        "adapt_deescalate_rate": args.adapt_deescalate_rate,
        "adapt_cooldown_s": args.adapt_cooldown,
        "adapt_censoring": args.adapt_censoring,
        "t_out_min": args.t_out_min,
        "t_out_max": args.t_out_max,
    }


def run_baseline(args, rows, run_dir, device):
    cfg = base_cfg(args, rows, run_dir)
    cfg["n_foreground"] = 0
    cfg["n_background"] = 0
    loader = baseline_loader(rows, cfg)
    consumer = Consumer(args.consumer, device)

    batches, warm = [], []
    consumer_wall = 0.0
    t_epoch = None
    t_last = time.perf_counter()
    for i, (tensors, orders, n_slow, durs) in enumerate(loader):
        if args.max_batches and i >= args.max_batches:
            break
        lat = time.perf_counter() - t_last
        c = consumer.step(tensors)
        consumer_wall += c
        rec = {"batch_index": i, "size": len(orders), "latency_s": lat,
               "mean_item_wait_s": (sum(durs) / len(durs)) if durs else 0.0,
               "n_truth_slow": n_slow, "n_route_slow": 0, "n_error": 0,
               "consumer_s": c}
        if i < args.warmup_batches:
            warm.append(rec)
            t_epoch = time.perf_counter()
        else:
            batches.append(rec)
        t_last = time.perf_counter()

    epoch_wall = time.perf_counter() - (t_epoch or t_last)
    write_batches(os.path.join(run_dir, "batches.csv"), batches)
    if warm:
        write_batches(os.path.join(run_dir, "batches_warmup.csv"), warm)
    measured = sum(b["size"] for b in batches)
    return {
        "epoch_wall_s": epoch_wall,
        "throughput_samples_s": measured / epoch_wall if epoch_wall > 0 else "",
        "consumer_wall_s": consumer_wall,
        "n_samples": len(rows),
    }


def run_minato(args, rows, run_dir, device, n_fg, n_bg):
    cfg = base_cfg(args, rows, run_dir)
    cfg["n_foreground"] = n_fg
    cfg["n_background"] = n_bg
    if args.pin:
        cfg["core_plan"] = plan_cores(n_fg, n_bg, args.front_threads)

    # --- warm-up profiling pass, over this run's own mixed manifest --------
    stages = pipelines.build_stages(args.pipeline,
                                    json.loads(args.pipeline_kwargs),
                                    seed=args.seed)
    calib = calibrate_t_out(stages, rows, args.warmup_samples,
                            args.t_out_percentile, args.seed,
                            margin=args.t_out_margin, discard=args.warmup_discard)
    cfg["t_out"] = calib["t_out"]
    with open(os.path.join(run_dir, "calibration.json"), "w") as fh:
        json.dump(calib, fh, indent=2, sort_keys=True)

    harness = MinatoHarness(rows, cfg,
                            trace_path=os.path.join(run_dir, "threshold_trace.csv"))
    consumer = Consumer(args.consumer, device)
    harness.start()

    batches, warm = [], []
    consumer_wall = 0.0
    t_epoch = time.perf_counter()
    for i, (batch, meta) in enumerate(harness.iter_batches()):
        tensors = [b["tensor"] for b in batch if b["tensor"] is not None]
        stacked = None
        if tensors:
            try:
                import torch
                stacked = torch.stack(tensors)
            except Exception:
                stacked = None
        c = consumer.step(stacked)
        consumer_wall += c
        meta["consumer_s"] = c
        if i < args.warmup_batches:
            warm.append(meta)
            t_epoch = time.perf_counter()
        else:
            batches.append(meta)

    epoch_wall = time.perf_counter() - t_epoch
    adapt = harness.adapt_summary()
    harness.shutdown()
    adapt = harness.adapt_summary() or adapt

    write_batches(os.path.join(run_dir, "batches.csv"), batches)
    if warm:
        write_batches(os.path.join(run_dir, "batches_warmup.csv"), warm)
    with open(os.path.join(run_dir, "queue_trace.json"), "w") as fh:
        json.dump(harness.queue_trace, fh)

    measured = sum(b["size"] for b in batches)
    received = getattr(harness, "received", measured)
    if received < len(rows):
        print("  WARNING: epoch ended with %d/%d samples received; the epoch "
              "timeout or a worker error truncated it"
              % (received, len(rows)))
    out = {
        "epoch_wall_s": epoch_wall,
        "throughput_samples_s": measured / epoch_wall if epoch_wall > 0 else "",
        "consumer_wall_s": consumer_wall,
        "n_samples": len(rows),
        "adapt": int(bool(args.adapt)),
        "t_out_s": calib["t_out"],
        "t_out_percentile": calib["percentile"],
        "warmup_n": calib["n_warmup"],
        "warmup_p50_s": calib["latency"]["p50"],
        "warmup_p95_s": calib["latency"]["p95"],
        "warmup_slow_below_t_out_frac": calib.get("warmup_slow_below_t_out_frac", ""),
        "stage_median_s": calib["stage_median_s"],
    }
    out.update(adapt)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pipeline", required=True,
                    choices=sorted(pipelines.REGISTRY))
    ap.add_argument("--pipeline-kwargs", dest="pipeline_kwargs", default="{}")
    ap.add_argument("--out-root", dest="out_root", default="results")
    ap.add_argument("--loaders", default="pytorch,minato")
    ap.add_argument("--splits", default="11F1B,9F3B,7F5B,5F7B,3F9B")
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    ap.add_argument("--baseline-workers", dest="baseline_workers", type=int,
                    default=12)
    ap.add_argument("--front-threads", dest="front_threads", type=int, default=1)
    ap.add_argument("--consumer", default="sleep:0.03",
                    help="none | sleep:<seconds> | model:<torchvision name>")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--warmup-samples", dest="warmup_samples", type=int,
                    default=200, help="samples timed in the t_out warm-up pass")
    ap.add_argument("--warmup-discard", dest="warmup_discard", type=int, default=5)
    ap.add_argument("--warmup-batches", dest="warmup_batches", type=int, default=5,
                    help="batches excluded from the measured epoch")
    ap.add_argument("--t-out-percentile", dest="t_out_percentile", type=float,
                    default=75.0)
    ap.add_argument("--t-out-margin", dest="t_out_margin", type=float, default=1.0)
    ap.add_argument("--fast-queue-size", dest="fast_queue_size", type=int,
                    default=64)
    ap.add_argument("--slow-drain-threshold", dest="slow_drain_threshold",
                    type=int, default=5)
    ap.add_argument("--queue-poll-s", dest="queue_poll_s", type=float, default=0.05)
    ap.add_argument("--queue-trace-interval", dest="queue_trace_interval",
                    type=float, default=0.25)
    ap.add_argument("--idle-timeout", dest="idle_timeout", type=float, default=60.0)
    ap.add_argument("--epoch-timeout", dest="epoch_timeout", type=float,
                    default=3600.0)
    ap.add_argument("--max-batches", dest="max_batches", type=int, default=0,
                    help="stop after this many MEASURED batches (0 = full epoch). "
                         "--warmup-batches are consumed first and not counted.")
    ap.add_argument("--t-out-min", dest="t_out_min", type=float,
                    default=ADAPT_DEFAULTS["t_out_min"])
    ap.add_argument("--t-out-max", dest="t_out_max", type=float,
                    default=ADAPT_DEFAULTS["t_out_max"])
    # ---- dynamic threshold adaptation -----------------------------------
    ap.add_argument("--adapt", action="store_true", default=False,
                    help="enable the background profiler that rewrites t_out "
                         "during the run. Off by default so runs stay "
                         "comparable with the fixed-t_out matrix.")
    ap.add_argument("--adapt-fallback-percentile", dest="adapt_fallback_percentile",
                    type=float, default=ADAPT_DEFAULTS["adapt_fallback_percentile"])
    ap.add_argument("--adapt-window", dest="adapt_window", type=int,
                    default=ADAPT_DEFAULTS["adapt_window"])
    ap.add_argument("--adapt-interval", dest="adapt_interval", type=float,
                    default=ADAPT_DEFAULTS["adapt_interval_s"])
    ap.add_argument("--adapt-min-samples", dest="adapt_min_samples", type=int,
                    default=ADAPT_DEFAULTS["adapt_min_samples"])
    ap.add_argument("--adapt-max-step-ratio", dest="adapt_max_step_ratio",
                    type=float, default=ADAPT_DEFAULTS["adapt_max_step_ratio"])
    ap.add_argument("--adapt-escalate-rate", dest="adapt_escalate_rate",
                    type=float, default=ADAPT_DEFAULTS["adapt_escalate_rate"])
    ap.add_argument("--adapt-deescalate-rate", dest="adapt_deescalate_rate",
                    type=float, default=ADAPT_DEFAULTS["adapt_deescalate_rate"])
    ap.add_argument("--adapt-cooldown", dest="adapt_cooldown", type=float,
                    default=ADAPT_DEFAULTS["adapt_cooldown_s"])
    ap.add_argument("--adapt-censoring", dest="adapt_censoring",
                    choices=["reconstructed", "censored"],
                    default=ADAPT_DEFAULTS["adapt_censoring"],
                    help="'censored' feeds the cut-off value of a migrated "
                         "sample back into the percentile, reproducing the "
                         "downward spiral described in hmp/adaptive.py")
    ap.add_argument("--pin", action="store_true", help="CPU affinity (Linux only)")
    ap.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = M.read_run_manifest(args.manifest)
    mmeta = M.read_profile_meta(args.manifest)
    fingerprint = M.manifest_fingerprint(rows)

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    os.makedirs(args.out_root, exist_ok=True)
    summary_csv = os.path.join(args.out_root, "summary.csv")
    stamp = time.strftime("%Y%m%d-%H%M%S")

    common = {
        "dataset": mmeta.get("dataset", ""),
        "pipeline": args.pipeline,
        "regime": mmeta.get("regime", ""),
        "slow_fraction": mmeta.get("slow_fraction_actual", ""),
        "batch_size": args.batch_size,
        "manifest_fingerprint": fingerprint,
        "seed": args.seed,
        "consumer": args.consumer,
    }

    jobs = []
    loaders = [x.strip() for x in args.loaders.split(",") if x.strip()]
    if "pytorch" in loaders:
        jobs.append(("pytorch", 0, 0))
    if "minato" in loaders:
        for s in args.splits.split(","):
            if s.strip():
                f, b = parse_split(s)
                jobs.append(("minato", f, b))

    print("manifest      : %s (%d samples, fingerprint %s)"
          % (args.manifest, len(rows), fingerprint))
    print("device        : %s   consumer: %s" % (device, args.consumer))
    print("runs          : %d" % len(jobs))

    for loader, n_fg, n_bg in jobs:
        label = loader if loader == "pytorch" else "%dF%dB" % (n_fg, n_bg)
        run_id = "%s_%s_%s%s" % (stamp, loader, label,
                                 ("_" + args.tag) if args.tag else "")
        run_dir = os.path.join(args.out_root, run_id)
        os.makedirs(os.path.join(run_dir, "events"), exist_ok=True)
        print("\n== %s ==" % run_id)

        monitor = SystemMonitor(os.path.join(run_dir, "system.csv"))
        monitor.start()
        t0 = time.time()
        try:
            if loader == "pytorch":
                result = run_baseline(args, rows, run_dir, device)
            else:
                result = run_minato(args, rows, run_dir, device, n_fg, n_bg)
        finally:
            monitor.stop()
            monitor.join(timeout=5)

        meta = dict(common)
        meta.update(result)
        meta.update({
            "run_id": run_id,
            "loader": loader,
            "n_foreground": n_fg,
            "n_background": n_bg,
            "baseline_workers": args.baseline_workers if loader == "pytorch" else "",
        })
        with open(os.path.join(run_dir, "run_meta.json"), "w") as fh:
            json.dump({**meta, "args": vars(args), "wall_s": time.time() - t0},
                      fh, indent=2, sort_keys=True, default=str)

        row = summarise(run_dir, meta)
        append_summary(summary_csv, row)
        print("  throughput  : %s samples/s" % row.get("throughput_samples_s"))
        print("  batch p95   : %s s" % row.get("batch_p95_s"))
        if loader == "minato":
            print("  t_out       : %.4fs   migrations: %s  (at stage 0: %s)"
                  % (result["t_out_s"], row.get("migrations"),
                     row.get("migrations_at_stage_0")))
            if args.adapt:
                print("  adaptation  : %s update(s), %s escalation(s) to P%s; "
                      "t_out %.4f -> %.4f (range %.4f..%.4f)"
                      % (result.get("adapt_updates"),
                         result.get("adapt_escalations"),
                         args.adapt_fallback_percentile,
                         result["t_out_s"], result.get("adapt_t_out_final", 0.0),
                         result.get("adapt_t_out_min", 0.0),
                         result.get("adapt_t_out_max", 0.0)))
            print("  slow recall : %s   misclassification: %s"
                  % (row.get("slow_recall"), row.get("misclassification_rate")))

    print("\nsummary appended to %s" % summary_csv)


if __name__ == "__main__":
    main()
