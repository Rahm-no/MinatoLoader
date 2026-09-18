#!/usr/bin/env python3
"""
Local smoke test for the two Minato changes. No GPU, no torchrun, no KiTS19.

It builds a small synthetic volume dataset with a deliberately heavy-tailed
transform, then checks:

  Phase 1  run_stages: resuming at stage k reproduces the full pipeline
           bit-for-bit, and an abort discards at most one stage.
  Phase 2  calibration: t_out really is the requested percentile of the
           profiled distribution, and the time/sample budgets are respected.
  Phase 3  end-to-end PytTrain: slow samples migrate mid-pipeline carrying
           their partial state, background workers resume at that index, and
           every sample comes out exactly once with the right shape/dtype.

Run from the Minato/ directory:

    python3 tools/smoke_test.py --profile-dir /tmp/minato_profile

"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import scipy.ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)           # .../Minato
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch.multiprocessing as tmp                                  # noqa: E402
from torchvision import transforms as tv_transforms                  # noqa: E402

from data_loading.minato_profiling import (                          # noqa: E402
    MinatoConfig, StageAborted, TimeoutCalibrator, build_stages, run_stages,
)
from data_loading import pytorch_loader                              # noqa: E402
from data_loading.pytorch_loader import (                            # noqa: E402
    Cast, GaussianNoise, PytTrain, RandBalancedCrop, RandFlip,
    RandomBrightnessAugmentation,
)

PASS = "  \033[32mPASS\033[0m  "
FAIL = "  \033[31mFAIL\033[0m  "
_FAILURES = []


WARN = "  \033[33mWARN\033[0m  "


def check(name, condition, detail="", soft=False):
    """soft=True: report but do not fail. Used for checks whose outcome depends
    on how much CPU the machine has, not on whether the code is correct."""
    suffix = ("  (%s)" % detail) if detail else ""
    if condition:
        print(PASS + name + suffix)
    elif soft:
        print(WARN + name + suffix)
    else:
        print(FAIL + name + suffix)
        _FAILURES.append(name)
    return bool(condition)


def cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# synthetic workload
# ---------------------------------------------------------------------------

class HeavyTailStage(object):
    """A transform with a heavy-tailed cost, like real decode/augment work.

    Must live at module scope so that it survives pickling under the 'spawn'
    start method (macOS default).
    """

    def __init__(self, seed=0, slow_prob=0.15, cheap_passes=1, heavy_passes=60,
                 sigma=1.1):
        self.seed = seed
        self.slow_prob = slow_prob
        self.cheap_passes = cheap_passes
        self.heavy_passes = heavy_passes
        self.sigma = sigma
        self._rng = None

    def __call__(self, data):
        if self._rng is None:
            self._rng = np.random.RandomState(
                (self.seed * 1000003 + os.getpid()) % (2 ** 31 - 1))
        if self._rng.rand() < self.slow_prob:
            passes = self.heavy_passes
        else:
            # real fast paths are not degenerate; give them some spread
            passes = self.cheap_passes + self._rng.randint(0, 4)
        image = np.array(data["image"], copy=True).astype(np.float32)
        for _ in range(passes):
            for c in range(image.shape[0]):
                image[c] = scipy.ndimage.gaussian_filter(image[c], sigma=self.sigma)
        data["image"] = image
        return data


class DeterministicStage(object):
    """Pure, deterministic stage used to prove resume correctness."""

    def __init__(self, k):
        self.k = k

    def __call__(self, data):
        data["image"] = data["image"] * 1.5 + self.k
        data["label"] = data["label"] + np.uint8(self.k % 3)
        return data


class SlowDeterministicStage(DeterministicStage):
    def __init__(self, k, seconds):
        super(SlowDeterministicStage, self).__init__(k)
        self.seconds = seconds

    def __call__(self, data):
        deadline = time.perf_counter() + self.seconds
        while time.perf_counter() < deadline:
            pass  # busy-wait: a sleep would not model CPU-bound work
        return super(SlowDeterministicStage, self).__call__(data)


def make_dataset(directory, n_cases, volume, seed=0):
    """Write case_XXXXX_x.npy / _y.npy pairs that look like the preprocessed
    KiTS19 layout: image (1, D, H, W) float32, label (1, D, H, W) uint8."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.RandomState(seed)
    d = h = w = volume
    x_paths, y_paths = [], []
    for i in range(n_cases):
        image = rng.rand(1, d, h, w).astype(np.float32)
        label = np.zeros((1, d, h, w), dtype=np.uint8)
        # one connected foreground blob, large enough for the crop logic
        lo, hi = d // 4, d // 4 + d // 2
        label[:, lo:hi, lo:hi, lo:hi] = 1
        xp = os.path.join(directory, "case_%05d_x.npy" % i)
        yp = os.path.join(directory, "case_%05d_y.npy" % i)
        np.save(xp, image)
        np.save(yp, label)
        x_paths.append(xp)
        y_paths.append(yp)
    return x_paths, y_paths


def build_smoke_transforms(seed=0, slow_prob=0.25, heavy_passes=14):
    return tv_transforms.Compose([
        RandFlip(),
        HeavyTailStage(seed=seed, slow_prob=slow_prob, heavy_passes=heavy_passes),
        Cast(types=(np.float32, np.uint8)),
        RandomBrightnessAugmentation(factor=0.3, prob=0.1),
        GaussianNoise(mean=0.0, std=0.1, prob=0.1),
    ])


# ---------------------------------------------------------------------------
# Phase 1 -- resume correctness
# ---------------------------------------------------------------------------

def phase_resume_semantics():
    print("\n--- Phase 1: resume-from-index semantics -------------------------")
    stages = [("s%d" % k, DeterministicStage(k)) for k in range(6)]
    base = {"image": np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4),
            "label": np.zeros((1, 4, 4, 4), dtype=np.uint8)}

    full = run_stages(stages, {k: v.copy() for k, v in base.items()})

    ok_all = True
    for k in range(len(stages) + 1):
        # run the first k stages with checkpointing, then resume from k
        sink = {}

        def checkpoint(stage, data, sink=sink):
            sink[stage] = data

        partial_stages = stages[:k]
        mid = run_stages(partial_stages, {kk: vv.copy() for kk, vv in base.items()},
                         checkpoint=checkpoint)
        resumed = run_stages(stages, mid, start_index=k)
        same = (np.array_equal(full["image"], resumed["image"]) and
                np.array_equal(full["label"], resumed["label"]))
        ok_all = ok_all and same
    check("resuming at every stage index reproduces the full pipeline", ok_all)

    # checkpoints must be private copies: a later in-place stage must not
    # corrupt an already-published checkpoint.
    class InPlace(object):
        def __call__(self, data):
            data["image"] += 100.0
            return data

    stages2 = [("a", DeterministicStage(1)), ("b", InPlace()), ("c", InPlace())]
    seen = {}
    run_stages(stages2, {k: v.copy() for k, v in base.items()},
               checkpoint=lambda s, d: seen.setdefault(s, d))
    expected_after_1 = base["image"] * 1.5 + 1
    check("checkpoint after stage 1 is not mutated by later in-place stages",
          np.array_equal(seen[1]["image"], expected_after_1))

    # abort semantics: exactly one stage of work is lost
    import threading
    abort = threading.Event()
    timings = []
    published = {}

    def cb(stage, data):
        published["stage"] = stage
        published["data"] = data
        if stage == 2:
            abort.set()          # migrate right after stage 2 completes

    try:
        run_stages(stages, {k: v.copy() for k, v in base.items()},
                   checkpoint=cb, abort_event=abort, timings=timings)
        aborted = False
        stage_at_abort = -1
    except StageAborted as exc:
        aborted = True
        stage_at_abort = exc.stage
    check("abort_event stops the pipeline at a stage boundary", aborted,
          "stopped at stage %d" % stage_at_abort)
    check("no stage runs after the abort is observed",
          len(timings) == 2 and published["stage"] == 2,
          "%d stage(s) executed, checkpoint at %s" % (len(timings), published.get("stage")))


# ---------------------------------------------------------------------------
# Phase 2 -- calibration
# ---------------------------------------------------------------------------

def phase_calibration(tmpdir):
    print("\n--- Phase 2: warm-up calibration --------------------------------")
    latencies = [0.01, 0.02, 0.02, 0.03, 0.05, 0.08, 0.10, 0.10, 0.12, 0.15,
                 0.18, 0.20, 0.25, 0.30, 0.40, 0.55, 0.70, 0.90, 1.30, 2.40]
    stages = [("s0", DeterministicStage(0)), ("s1", DeterministicStage(1))]

    calls = {"n": 0}

    def measure(idx):
        calls["n"] += 1
        total = latencies[idx % len(latencies)]
        return total, [("s0", total * 0.4), ("s1", total * 0.6)]

    cache = os.path.join(tmpdir, "calib.json")
    cfg = MinatoConfig(calibrate=True, calib_seconds=30.0, calib_max_samples=100,
                       calib_min_samples=5, calib_warmup_samples=0,
                       calib_percentile=75.0, calib_cache=cache, profile_dir="")
    res = TimeoutCalibrator(cfg, stages, tag="unit").calibrate(
        measure, n_dataset=len(latencies), seed=7, log=lambda *a: None)

    expected = float(np.percentile(np.asarray(latencies), 75.0))
    check("t_out equals P75 of the profiled distribution",
          abs(res["t_out"] - expected) < 1e-9,
          "got %.4fs, expected %.4fs" % (res["t_out"], expected))
    check("t_out is not the hard-coded 2.0s", abs(res["t_out"] - 2.0) > 1e-6,
          "t_out=%.4fs" % res["t_out"])
    check("percentile is configurable",
          abs(TimeoutCalibrator(
              MinatoConfig(calib_percentile=95.0, calib_warmup_samples=0,
                           calib_min_samples=5, calib_max_samples=100,
                           calib_cache="", profile_dir=""),
              stages).calibrate(measure, len(latencies), seed=7,
                                log=lambda *a: None)["t_out"]
              - float(np.percentile(np.asarray(latencies), 95.0))) < 1e-9)

    n_first = calls["n"]
    res2 = TimeoutCalibrator(cfg, stages, tag="unit").calibrate(
        measure, n_dataset=len(latencies), seed=7, log=lambda *a: None)
    check("calibration result is cached and reused",
          res2.get("from_cache") is True and calls["n"] == n_first,
          "%d extra measurements" % (calls["n"] - n_first))

    # sample budget is honoured
    cfg3 = MinatoConfig(calib_seconds=3600.0, calib_max_samples=11,
                        calib_min_samples=1, calib_warmup_samples=0,
                        calib_cache="", profile_dir="")
    before = calls["n"]
    TimeoutCalibrator(cfg3, stages).calibrate(measure, 1000, seed=1,
                                              log=lambda *a: None)
    check("sample budget stops the pass", calls["n"] - before == 11,
          "%d samples measured" % (calls["n"] - before))

    # time budget is honoured
    def slow_measure(idx):
        time.sleep(0.05)
        return 0.05, [("s0", 0.05)]

    cfg4 = MinatoConfig(calib_seconds=0.4, calib_max_samples=10000,
                        calib_min_samples=2, calib_warmup_samples=0,
                        calib_cache="", profile_dir="")
    t0 = time.perf_counter()
    r4 = TimeoutCalibrator(cfg4, stages).calibrate(slow_measure, 10000, seed=1,
                                                   log=lambda *a: None)
    wall = time.perf_counter() - t0
    check("time budget stops the pass", wall < 2.0 and r4["latency"]["n"] >= 2,
          "%.2fs wall, %d samples" % (wall, r4["latency"]["n"]))

    cfg5 = MinatoConfig(calibrate=False, sample_timeout=1.25, calib_cache="",
                        profile_dir="")
    r5 = TimeoutCalibrator(cfg5, stages).calibrate(measure, 10, seed=1,
                                                   log=lambda *a: None)
    check("--calibrate false falls back to the fixed threshold",
          abs(r5["t_out"] - 1.25) < 1e-9 and r5["source"] == "fallback")


# ---------------------------------------------------------------------------
# Phase 3 -- end-to-end dataset
# ---------------------------------------------------------------------------

def phase_migration_protocol(args, data_dir, profile_dir):
    """Deterministic: prove the payload and the resume index are correct.

    One stage is rigged to always blow the budget, so every sample migrates at
    a known index.  Nothing here is CPU-sensitive, so the result does not
    depend on how many cores the machine has.
    """
    print("\n--- Phase 3: migration payload & resume protocol -----------------")
    x, y = make_dataset(os.path.join(data_dir, "proto"), 6, 48, seed=5)

    slow_stage_seconds = 0.30
    t_out = 0.10
    pipeline = tv_transforms.Compose([
        DeterministicStage(1),
        SlowDeterministicStage(2, slow_stage_seconds),
        DeterministicStage(3),
        DeterministicStage(4),
    ])

    cfg = MinatoConfig(calibrate=False, sample_timeout=t_out, t_out_min=0.001,
                       calib_cache="", profile_dir="", num_slow_workers=0,
                       front_threads=1, verbose=args.verbose)

    slow_queue = tmp.Queue()
    processed_queue = tmp.Queue()
    dataset = PytTrain(x, y, slow_queue, processed_queue,
                       train_transforms=pipeline, patch_size=[32] * 3,
                       oversampling=0.4, minato_cfg=cfg)

    check("no background workers were started with num_slow_workers=0",
          len(dataset.slow_workers) == 0)
    check("fallback threshold is used when calibration is off",
          abs(dataset.sample_timeout - t_out) < 1e-9,
          "t_out=%.3fs" % dataset.sample_timeout)

    outs = [dataset[i] for i in range(len(dataset))]
    check("every sample with a stage over budget was migrated",
          all(o is None for o in outs),
          "%d/%d migrated" % (sum(o is None for o in outs), len(outs)))

    payloads = []
    deadline = time.perf_counter() + 30
    while len(payloads) < len(outs) and time.perf_counter() < deadline:
        try:
            payloads.append(slow_queue.get(timeout=0.5))
        except Exception:
            break
    check("one migration payload per migrated sample",
          len(payloads) == len(outs), "%d payloads" % len(payloads))

    shaped = all(isinstance(p, tuple) and len(p) == 4 for p in payloads)
    check("payload is (idx, resume_stage, image, label)", shaped)

    # stages = [RandBalancedCrop, Det1, SlowDet2, Det3, Det4]; the budget is
    # blown inside SlowDet2, so two stages are complete when we migrate.
    stages_at = sorted(set(p[1] for p in payloads)) if shaped else []
    check("resume index points at the interrupted transformation",
          stages_at == [2], "resume stages observed: %s" % stages_at)
    check("partial state travels with the sample, not just the index",
          shaped and all(p[2] is not None and p[3] is not None for p in payloads))
    check("partial state is already cropped to the patch size",
          shaped and all(tuple(p[2].shape[1:]) == (32, 32, 32) for p in payloads),
          "shapes: %s" % sorted(set(tuple(p[2].shape) for p in payloads)))

    # hand the payloads to a real background worker and check the output
    for p in payloads:
        slow_queue.put(p)
    slow_queue.put(None)
    pytorch_loader.slow_worker(dataset.images_data, dataset.labels_data,
                               [32] * 3, 0.4, pipeline, slow_queue,
                               processed_queue, cfg, 0)

    results = []
    while len(results) < len(payloads):
        try:
            results.append(processed_queue.get(timeout=5))
        except Exception:
            break
    check("background worker emitted one result per payload",
          len(results) == len(payloads), "%d results" % len(results))

    # image_final = ((P*1.5 + 2)*1.5 + 3)*1.5 + 4   -- exactly, in float32
    exact = True
    for (idx, stage, image, label), (out_image, out_label) in zip(payloads, results):
        expected = ((image * 1.5 + 2) * 1.5 + 3) * 1.5 + 4
        expected_label = label + np.uint8(2 % 3) + np.uint8(3 % 3) + np.uint8(4 % 3)
        exact = exact and np.array_equal(out_image, expected) \
            and np.array_equal(out_label, expected_label)
    check("resumed output equals the remaining stages applied to the payload",
          exact and bool(results))

    dataset.shutdown()


def phase_end_to_end(args, data_dir, profile_dir):
    print("\n--- Phase 4: heavy-tailed workload, end to end -------------------")
    x, y = make_dataset(data_dir, args.cases, args.volume, seed=3)

    cfg = MinatoConfig(
        calibrate=True,
        calib_seconds=args.calib_seconds,
        calib_max_samples=args.calib_max_samples,
        calib_min_samples=4,
        calib_warmup_samples=1,
        calib_percentile=args.percentile,
        calib_margin=args.margin,
        calib_cache="",
        t_out_min=0.001,
        profile_dir=profile_dir,
        profile_fast_samples=True,
        num_slow_workers=args.slow_workers,
        front_threads=args.front_threads,
        verbose=args.verbose,
    )

    slow_queue = tmp.Queue()
    slow_processed_queue = tmp.Queue()

    t0 = time.perf_counter()
    dataset = PytTrain(
        x, y, slow_queue, slow_processed_queue,
        train_transforms=build_smoke_transforms(
            seed=11, slow_prob=args.slow_prob, heavy_passes=args.heavy_passes),
        patch_size=[args.patch] * 3, oversampling=0.4, minato_cfg=cfg)
    calib_wall = time.perf_counter() - t0

    calib = dataset.calibration
    print("    stages            : %s" % " -> ".join(calib["stages"]))
    print("    t_out             : %.4fs  (P%.0f)" % (dataset.sample_timeout,
                                                      calib["percentile"]))
    print("    profiled          : %d samples in %.1fs"
          % (calib["latency"]["n"], calib_wall))
    print("    latency p50/p75/p90/max: %.4f / %.4f / %.4f / %.4f"
          % (calib["latency"]["p50"], calib["latency"]["p75"],
             calib["latency"]["p90"], calib["latency"]["max"]))

    check("calibration produced a workload-relative t_out",
          calib["source"].startswith("profiled") and calib["latency"]["n"] >= 4)
    check("t_out sits inside the observed latency range",
          calib["latency"]["min"] <= dataset.sample_timeout <= calib["latency"]["max"],
          "%.4f in [%.4f, %.4f]" % (dataset.sample_timeout,
                                    calib["latency"]["min"], calib["latency"]["max"]))
    check("pipeline is stage-indexed", len(dataset.stages) >= 3,
          "%d stages" % len(dataset.stages))

    # ---- drive the front path -------------------------------------------
    fast, migrated = [], []
    t0 = time.perf_counter()
    for epoch_idx in range(args.passes):
        for idx in range(len(dataset)):
            out = dataset[idx]
            if out is None:
                migrated.append(idx)
            else:
                fast.append((idx, out))
    front_wall = time.perf_counter() - t0
    total = args.passes * len(dataset)
    print("    front path        : %d fast, %d migrated, in %.1fs"
          % (len(fast), len(migrated), front_wall))

    if not migrated:
        print("    hint: nothing exceeded t_out on this machine; rerun with a "
              "heavier tail, e.g. --heavy-passes %d --slow-prob 0.3"
              % (args.heavy_passes * 3))
    check("some samples exceeded t_out and were migrated", len(migrated) > 0,
          "%d/%d migrated" % (len(migrated), total))
    rate = len(migrated) / float(total)
    check("the fast path still completes samples", len(fast) > 0,
          "%d/%d fast, migration rate %.0f%%" % (len(fast), total, 100 * rate),
          soft=True)
    if rate > 0.6:
        print("    note: migration rate is high. P75 migrates ~25% of samples by "
              "definition; abandoned stages then steal CPU from the front")
        print("          worker and amplify that. Raise --calib_percentile or "
              "--calib_margin, or give the pool more --front_threads.")

    # ---- collect resumed samples ----------------------------------------
    collected = []
    deadline = time.perf_counter() + args.collect_timeout
    while len(collected) < len(migrated) and time.perf_counter() < deadline:
        try:
            collected.append(slow_processed_queue.get(timeout=0.25))
        except Exception:
            pass
    check("every migrated sample came back from a background worker",
          len(collected) == len(migrated),
          "%d/%d recovered" % (len(collected), len(migrated)))

    ref_image, ref_label = fast[0][1] if fast else (None, None)
    shapes_ok = dtypes_ok = True
    for image, label in collected:
        if ref_image is not None:
            shapes_ok = shapes_ok and image.shape == ref_image.shape \
                and label.shape == ref_label.shape
            dtypes_ok = dtypes_ok and image.dtype == ref_image.dtype \
                and label.dtype == ref_label.dtype
        shapes_ok = shapes_ok and tuple(image.shape[1:]) == (args.patch,) * 3
        finite = np.isfinite(image).all()
        dtypes_ok = dtypes_ok and finite
    check("resumed samples have the same shape as fast-path samples", shapes_ok)
    check("resumed samples have the same dtype and are finite", dtypes_ok)

    dataset.shutdown()

    # ---- the actual claims ----------------------------------------------
    time.sleep(0.5)  # let background recorders flush
    from tools.minato_report import build_report
    rep = build_report(profile_dir)

    at_stage = {int(k): v for k, v in rep["migrations_at_stage"].items()}
    with_partial = sum(v for k, v in at_stage.items() if k >= 1)
    n_mig = max(1, sum(at_stage.values()))
    # (proved deterministically in phase 3; here it is CPU-contention sensitive)
    check("most migrations still carry partial state under load",
          with_partial / float(n_mig) >= 0.7,
          "%d/%d resumed at stage >= 1" % (with_partial, n_mig), soft=True)

    waste = rep.get("mean_wasted_per_migration_s", 0.0)
    pipeline = rep["pipeline_median_cost_s"]
    check("wasted CPU per migration is below the cost of a whole pipeline",
          0.0 < waste < pipeline,
          "%.4fs wasted vs %.4fs full pipeline" % (waste, pipeline), soft=True)
    check("resume-from-index wastes less than the original full restart",
          rep["wasted_cpu_resume_from_index_s"] < rep["wasted_cpu_full_restart_s"],
          "%.3fs vs %.3fs" % (rep["wasted_cpu_resume_from_index_s"],
                              rep["wasted_cpu_full_restart_s"]))
    check("background workers skipped work inherited from the front worker",
          rep["background_work_skipped_s"] > 0.0,
          "%.3fs of stages not re-executed" % rep["background_work_skipped_s"])

    return {"fast": len(fast), "migrated": len(migrated),
            "recovered": len(collected), "t_out": dataset.sample_timeout}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", default=os.path.join(tempfile.gettempdir(),
                                                          "minato_profile"))
    ap.add_argument("--keep", action="store_true", help="keep the synthetic dataset")
    ap.add_argument("--cases", type=int, default=16)
    ap.add_argument("--volume", type=int, default=96, help="synthetic volume edge")
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--passes", type=int, default=2, help="sweeps over the dataset")
    ap.add_argument("--slow-prob", dest="slow_prob", type=float, default=0.15)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--heavy-passes", dest="heavy_passes", type=int, default=60)
    ap.add_argument("--percentile", type=float, default=75.0)
    ap.add_argument("--calib-seconds", dest="calib_seconds", type=float, default=25.0)
    ap.add_argument("--calib-max-samples", dest="calib_max_samples", type=int, default=40)
    ap.add_argument("--slow-workers", dest="slow_workers", type=int, default=0,
                    help="0 = auto-size from the CPU count")
    ap.add_argument("--front-threads", dest="front_threads", type=int, default=2)
    ap.add_argument("--collect-timeout", dest="collect_timeout", type=float, default=120.0)
    ap.add_argument("--start-method", dest="start_method", default="spawn",
                    choices=["spawn", "fork", "forkserver"],
                    help="'fork' mirrors the Linux/Docker path; 'spawn' is the "
                         "macOS default and is the safer local choice")
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    try:
        tmp.set_start_method(args.start_method, force=True)
    except RuntimeError:
        pass

    ncpu = cpu_count()
    if args.slow_workers <= 0:
        args.slow_workers = max(1, min(3, ncpu - 2))

    print("Minato smoke test")
    print("  python           : %s" % sys.version.split()[0])
    print("  usable CPUs      : %d  (slow workers: %d, front threads: %d)"
          % (ncpu, args.slow_workers, args.front_threads))
    print("  numpy / scipy    : %s / %s" % (np.__version__, scipy.__version__))
    print("  start method     : %s" % tmp.get_start_method())
    print("  profile dir      : %s" % args.profile_dir)

    if os.path.isdir(args.profile_dir):
        shutil.rmtree(args.profile_dir, ignore_errors=True)
    os.makedirs(args.profile_dir, exist_ok=True)

    workdir = tempfile.mkdtemp(prefix="minato_smoke_")
    try:
        phase_resume_semantics()
        phase_calibration(workdir)
        phase_migration_protocol(args, os.path.join(workdir, "data"),
                                 args.profile_dir)
        if not args.skip_e2e:
            if ncpu < 4:
                print("\n  note: only %d usable CPU(s). Phase 4 shares them between "
                      "the front worker," % ncpu)
                print("        the abandoned stages and the background workers, so "
                      "the migration rate")
                print("        will look much worse than on a real node. "
                      "Correctness is covered by phases 1-3.")
            phase_end_to_end(args, os.path.join(workdir, "data"), args.profile_dir)
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print("\nkept synthetic dataset in %s" % workdir)

    if not args.skip_e2e:
        print("\n--- Report -------------------------------------------------------")
        from tools.minato_report import build_report, print_report
        print_report(build_report(args.profile_dir))

    print("=" * 66)
    if _FAILURES:
        print("FAILED: %d check(s): %s" % (len(_FAILURES), ", ".join(_FAILURES)))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
