"""
Minato: workload-relative timeout calibration + resume-from-index migration.

This module is deliberately free of any torch / torchvision import so that it
can be unit-tested and profiled on a laptop without a CUDA stack.

Two ideas live here:

1. ``TimeoutCalibrator`` -- a warm-up profiling pass that measures the
   end-to-end preprocessing latency of a random subset of the *actual*
   workload and sets ``t_out`` to a percentile (P75 by default) of that
   distribution.  "Slow" is therefore defined relative to the observed
   distribution instead of a hard-coded 2.0 s constant.

2. ``run_stages`` -- the preprocessing pipeline is executed as an *indexed*
   list of stages, publishing a checkpoint after each stage.  When the front
   worker decides to migrate a sample, it ships ``(idx, stage_index, image,
   label)`` to the background workers, which resume at ``stage_index``.  The
   only work thrown away is the single stage that was in flight.
"""

import csv
import json
import os
import platform
import random
import socket
import threading
import time

import numpy as np

SCHEMA_VERSION = 2

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def _env(name, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        if cast is bool:
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return cast(raw)
    except (TypeError, ValueError):
        return default


class MinatoConfig(object):
    """All knobs for calibration + migration.

    Precedence: explicit kwarg  >  environment variable  >  default.
    """

    FIELDS = {
        # ---- calibration -------------------------------------------------
        "calibrate": (True, bool, "MINATO_CALIBRATE"),
        "calib_seconds": (600.0, float, "MINATO_CALIB_SECONDS"),
        "calib_max_samples": (256, int, "MINATO_CALIB_MAX_SAMPLES"),
        "calib_min_samples": (24, int, "MINATO_CALIB_MIN_SAMPLES"),
        "calib_warmup_samples": (3, int, "MINATO_CALIB_WARMUP_SAMPLES"),
        "calib_percentile": (75.0, float, "MINATO_CALIB_PERCENTILE"),
        "calib_margin": (1.0, float, "MINATO_CALIB_MARGIN"),
        "calib_cache": ("", str, "MINATO_CALIB_CACHE"),
        "recalibrate": (False, bool, "MINATO_RECALIBRATE"),
        # hard safety rails on the calibrated value
        "t_out_min": (0.05, float, "MINATO_TOUT_MIN"),
        "t_out_max": (120.0, float, "MINATO_TOUT_MAX"),
        # fallback when calibration is disabled or produced too few samples
        "sample_timeout": (2.0, float, "MINATO_SAMPLE_TIMEOUT"),
        # ---- dynamic threshold adaptation (paper 4.2) ---------------------
        # Off by default: enabling it means t_out is no longer a fixed
        # experimental variable, which changes what a run measures.
        "adapt": (False, bool, "MINATO_ADAPT"),
        "adapt_fallback_percentile": (90.0, float, "MINATO_ADAPT_FALLBACK"),
        "adapt_window": (512, int, "MINATO_ADAPT_WINDOW"),
        "adapt_interval_s": (5.0, float, "MINATO_ADAPT_INTERVAL"),
        "adapt_min_samples": (64, int, "MINATO_ADAPT_MIN_SAMPLES"),
        "adapt_max_step_ratio": (2.0, float, "MINATO_ADAPT_MAX_STEP"),
        "adapt_escalate_rate": (0.35, float, "MINATO_ADAPT_ESCALATE"),
        "adapt_deescalate_rate": (0.20, float, "MINATO_ADAPT_DEESCALATE"),
        "adapt_cooldown_s": (20.0, float, "MINATO_ADAPT_COOLDOWN"),
        "adapt_censoring": ("reconstructed", str, "MINATO_ADAPT_CENSORING"),
        # ---- migration ---------------------------------------------------
        "checkpoint_copy": (True, bool, "MINATO_CHECKPOINT_COPY"),
        "front_threads": (2, int, "MINATO_FRONT_THREADS"),
        # 0 leaves the slow queue unserviced: useful for tests that want to
        # inspect the migration payload before anything consumes it.
        "num_slow_workers": (3, int, "MINATO_SLOW_WORKERS"),
        "queue_wait_timeout": (30.0, float, "MINATO_QUEUE_WAIT_TIMEOUT"),
        # ---- instrumentation ---------------------------------------------
        "profile_dir": ("", str, "MINATO_PROFILE_DIR"),
        "profile_fast_samples": (False, bool, "MINATO_PROFILE_FAST"),
        "verbose": (False, bool, "MINATO_VERBOSE"),
    }

    def __init__(self, **kwargs):
        for name, (default, cast, envvar) in self.FIELDS.items():
            if name in kwargs and kwargs[name] is not None:
                value = kwargs[name]
                if cast is bool and not isinstance(value, bool):
                    value = str(value).strip().lower() in ("1", "true", "yes", "on")
                else:
                    value = cast(value)
            else:
                value = _env(envvar, default, cast)
            setattr(self, name, value)

    @classmethod
    def from_flags(cls, flags):
        picked = {}
        for name in cls.FIELDS:
            if hasattr(flags, name):
                picked[name] = getattr(flags, name)
        return cls(**picked)

    def as_dict(self):
        return {name: getattr(self, name) for name in self.FIELDS}

    def __repr__(self):
        body = ", ".join("%s=%r" % (k, v) for k, v in sorted(self.as_dict().items()))
        return "MinatoConfig(%s)" % body


# --------------------------------------------------------------------------
# indexed pipeline execution
# --------------------------------------------------------------------------


class StageAborted(Exception):
    """Raised inside the front-worker thread once the sample has been migrated.

    ``stage`` is the number of stages that completed before the abort.
    """

    def __init__(self, stage):
        super(StageAborted, self).__init__("aborted after stage %d" % stage)
        self.stage = stage


def build_stages(rand_crop, train_transforms):
    """Return the ordered ``[(name, callable), ...]`` pipeline.

    The position in this list *is* the resume index that travels with a
    migrated sample.  Stage 0 is the balanced random crop, stages 1..N-1 are
    the entries of the ``transforms.Compose``.
    """
    stages = []
    if rand_crop is not None:
        stages.append((type(rand_crop).__name__, rand_crop))
    seq = getattr(train_transforms, "transforms", train_transforms) or []
    for t in seq:
        stages.append((type(t).__name__, t))
    return stages


def stage_names(stages):
    return [name for name, _ in stages]


def clone_data(data):
    """Private deep-ish copy of a ``{"image", "label"}`` sample.

    Several transforms (e.g. HeavyGaussianBlur) write into their input array,
    so a checkpoint must own its memory; otherwise the still-running stage
    would corrupt the state we are about to migrate.  Also materialises numpy
    memmaps into RAM, which is what we want before shipping over a queue.
    """
    out = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) or isinstance(value, np.memmap):
            out[key] = np.array(value, copy=True)
        else:
            out[key] = value
    return out


def run_stages(stages, data, start_index=0, checkpoint=None, abort_event=None,
               timings=None, copy_checkpoints=True):
    """Execute ``stages[start_index:]`` on ``data``.

    checkpoint       callable(stages_completed, data) invoked after every
                     stage with a private copy of the intermediate sample.
                     ``None`` disables checkpointing (background/resume path,
                     where nothing can be migrated again).
    abort_event      threading.Event checked at every stage boundary.  When
                     set, ``StageAborted`` is raised, so at most one stage of
                     work is discarded.
    timings          optional list; ``(stage_name, seconds)`` is appended per
                     stage.
    copy_checkpoints when False, checkpoints alias the live arrays.  Cheaper,
                     but only safe if no stage mutates its input in place.
                     Exposed so the copy overhead can be measured.
    """
    n = len(stages)
    for i in range(start_index, n):
        if abort_event is not None and abort_event.is_set():
            raise StageAborted(i)
        name, fn = stages[i]
        t0 = time.perf_counter()
        data = fn(data)
        if timings is not None:
            timings.append((name, time.perf_counter() - t0))
        if checkpoint is not None:
            if abort_event is not None and abort_event.is_set():
                # The coordinator already gave up on us; do not publish, the
                # migration payload is the previous checkpoint.
                raise StageAborted(i)
            checkpoint(i + 1, clone_data(data) if copy_checkpoints else dict(data))
    return data


# --------------------------------------------------------------------------
# warm-up calibration
# --------------------------------------------------------------------------


def _percentiles(values):
    arr = np.asarray(values, dtype=np.float64)
    keys = [("p01", 1), ("p25", 25), ("p50", 50), ("p75", 75),
            ("p90", 90), ("p95", 95), ("p99", 99)]
    out = {"n": int(arr.size),
           "min": float(arr.min()),
           "max": float(arr.max()),
           "mean": float(arr.mean()),
           "std": float(arr.std())}
    for key, q in keys:
        out[key] = float(np.percentile(arr, q))
    return out


class TimeoutCalibrator(object):
    """Warm-up profiling pass that turns a latency distribution into t_out."""

    def __init__(self, cfg, stages, tag=""):
        self.cfg = cfg
        self.stages = stages
        self.tag = tag

    # -- cache ------------------------------------------------------------
    def _fingerprint(self, n_dataset, extra):
        payload = {
            "schema": SCHEMA_VERSION,
            "stages": stage_names(self.stages),
            "percentile": self.cfg.calib_percentile,
            "margin": self.cfg.calib_margin,
            "checkpoint_copy": bool(self.cfg.checkpoint_copy),
            "n_dataset": int(n_dataset),
            "machine": platform.machine(),
            "extra": extra,
        }
        return json.dumps(payload, sort_keys=True)

    def load_cache(self, fingerprint):
        path = self.cfg.calib_cache
        if not path or self.cfg.recalibrate or not os.path.exists(path):
            return None
        try:
            with open(path, "r") as fh:
                blob = json.load(fh)
        except (ValueError, OSError):
            return None
        if blob.get("fingerprint") != fingerprint:
            return None
        return blob

    def save_cache(self, result):
        path = self.cfg.calib_cache
        if not path:
            return
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError:
                return
        tmp = "%s.tmp.%d" % (path, os.getpid())
        try:
            with open(tmp, "w") as fh:
                json.dump(result, fh, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except OSError:
            pass

    # -- the pass ---------------------------------------------------------
    def calibrate(self, measure_fn, n_dataset, seed=0, extra=None, log=print):
        """measure_fn(idx) -> (total_seconds, [(stage_name, seconds), ...])"""
        cfg = self.cfg
        fingerprint = self._fingerprint(n_dataset, extra or {})

        cached = self.load_cache(fingerprint)
        if cached is not None:
            log("[minato/calib] reusing cached calibration from %s: "
                "t_out=%.4fs (P%.0f over %d samples)"
                % (cfg.calib_cache, cached["t_out"], cached["percentile"],
                   cached["latency"]["n"]))
            cached["from_cache"] = True
            return cached

        if not cfg.calibrate:
            return self._fallback(fingerprint, "calibration disabled", log)

        rng = random.Random(seed)
        order = list(range(n_dataset))
        rng.shuffle(order)

        budget = cfg.calib_seconds
        log("[minato/calib] warm-up profiling pass: <=%.0fs / <=%d samples, "
            "stages=%s" % (budget, cfg.calib_max_samples, stage_names(self.stages)))

        totals = []
        per_stage = {}
        t_start = time.perf_counter()
        attempted = 0
        failed = 0
        i = 0
        while True:
            elapsed = time.perf_counter() - t_start
            if len(totals) >= cfg.calib_max_samples:
                stop = "sample budget"
                break
            if elapsed >= budget and len(totals) >= cfg.calib_min_samples:
                stop = "time budget"
                break
            if elapsed >= budget and attempted >= n_dataset:
                stop = "time budget (dataset exhausted)"
                break
            if attempted >= max(n_dataset, cfg.calib_max_samples) and len(totals) > 0:
                stop = "dataset exhausted"
                break
            idx = order[i % len(order)]
            i += 1
            attempted += 1
            try:
                total, timings = measure_fn(idx)
            except Exception as exc:  # a bad volume must not kill the run
                failed += 1
                log("[minato/calib] sample %d failed: %r" % (idx, exc))
                continue
            # discard the first few: cold page cache / allocator warm-up
            if attempted <= cfg.calib_warmup_samples:
                continue
            totals.append(total)
            for name, dt in timings:
                per_stage.setdefault(name, []).append(dt)
            if cfg.verbose:
                log("[minato/calib]   idx=%d total=%.4fs" % (idx, total))

        wall = time.perf_counter() - t_start
        if len(totals) < max(2, cfg.calib_min_samples // 4):
            return self._fallback(
                fingerprint,
                "only %d usable samples in %.1fs (%d failures)" % (len(totals), wall, failed),
                log)

        latency = _percentiles(totals)
        raw = float(np.percentile(np.asarray(totals, dtype=np.float64),
                                  cfg.calib_percentile)) * cfg.calib_margin
        t_out = min(max(raw, cfg.t_out_min), cfg.t_out_max)

        stage_stats = {}
        for name, values in per_stage.items():
            stage_stats[name] = _percentiles(values)

        result = {
            "schema": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "tag": self.tag,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "created": time.time(),
            "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "profiled",
            "stop_reason": stop,
            "calibration_wall_seconds": wall,
            "attempted": attempted,
            "failed": failed,
            "percentile": cfg.calib_percentile,
            "margin": cfg.calib_margin,
            "t_out_raw": raw,
            "t_out": t_out,
            "clamped": bool(t_out != raw),
            "stages": stage_names(self.stages),
            "latency": latency,
            "stage_latency": stage_stats,
            "samples": totals,
            "config": self.cfg.as_dict(),
            "from_cache": False,
        }
        log("[minato/calib] done in %.1fs over %d samples -> t_out = %.4fs "
            "(P%.0f x %.2f; P50=%.4f P90=%.4f max=%.4f)"
            % (wall, latency["n"], t_out, cfg.calib_percentile, cfg.calib_margin,
               latency["p50"], latency["p90"], latency["max"]))
        self.save_cache(result)
        return result

    def _fallback(self, fingerprint, reason, log):
        t_out = min(max(self.cfg.sample_timeout, self.cfg.t_out_min), self.cfg.t_out_max)
        log("[minato/calib] falling back to fixed t_out=%.4fs (%s)" % (t_out, reason))
        return {
            "schema": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "tag": self.tag,
            "source": "fallback",
            "reason": reason,
            "created": time.time(),
            "percentile": self.cfg.calib_percentile,
            "t_out": t_out,
            "t_out_raw": t_out,
            "stages": stage_names(self.stages),
            "latency": {"n": 0},
            "stage_latency": {},
            "samples": [],
            "config": self.cfg.as_dict(),
            "from_cache": False,
        }


# --------------------------------------------------------------------------
# instrumentation
# --------------------------------------------------------------------------

RECORD_FIELDS = [
    "ts", "role", "pid", "event", "idx",
    "stage", "n_stages", "queue_wait_s", "elapsed_s",
    "preserved_s", "wasted_s", "stage_times",
]


class MigrationRecorder(object):
    """Append-only CSV, one file per process (no cross-process locking)."""

    def __init__(self, profile_dir, role, enabled=True):
        self.profile_dir = profile_dir or ""
        self.role = role
        self.enabled = bool(enabled and self.profile_dir)
        self._pid = None
        self._fh = None
        self._writer = None
        self._lock = threading.Lock()
        self.counters = {}

    def _bump(self, key, value=1):
        self.counters[key] = self.counters.get(key, 0) + value

    def _ensure(self):
        if not self.enabled:
            return False
        pid = os.getpid()
        if self._fh is not None and self._pid == pid:
            return True
        # new process (post-fork): open our own file
        self._pid = pid
        try:
            os.makedirs(self.profile_dir, exist_ok=True)
            path = os.path.join(self.profile_dir,
                                "events_%s_%d.csv" % (self.role, pid))
            new = not os.path.exists(path)
            self._fh = open(path, "a", newline="")
            self._writer = csv.writer(self._fh)
            if new:
                self._writer.writerow(RECORD_FIELDS)
                self._fh.flush()
        except OSError:
            self.enabled = False
            return False
        return True

    def write(self, event, idx, stage=-1, n_stages=-1, queue_wait_s=0.0,
              elapsed_s=0.0, preserved_s=0.0, wasted_s=0.0, stage_times=None):
        self._bump(event)
        self._bump(event + "_seconds", elapsed_s)
        if not self.enabled:
            return
        with self._lock:
            if not self._ensure():
                return
            try:
                self._writer.writerow([
                    "%.6f" % time.time(), self.role, os.getpid(), event, idx,
                    stage, n_stages, "%.6f" % queue_wait_s, "%.6f" % elapsed_s,
                    "%.6f" % preserved_s, "%.6f" % wasted_s,
                    json.dumps([[n, round(d, 6)] for n, d in (stage_times or [])]),
                ])
                self._fh.flush()
            except (OSError, ValueError):
                self.enabled = False

    def dump_calibration(self, result, tag):
        if not self.profile_dir:
            return
        try:
            os.makedirs(self.profile_dir, exist_ok=True)
            path = os.path.join(self.profile_dir,
                                "calibration_%s_%d.json" % (tag, os.getpid()))
            with open(path, "w") as fh:
                json.dump(result, fh, indent=2, sort_keys=True)
        except (OSError, TypeError):
            pass

    def close(self):
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None
                self._writer = None


# --------------------------------------------------------------------------
# dynamic threshold adaptation
# --------------------------------------------------------------------------

class SharedThreshold(object):
    """t_out in shared memory, so every forked producer sees an update.

    Created with the stdlib multiprocessing primitives that the rest of this
    loader uses (see the note about torch.multiprocessing being shadowed in
    pytorch_loader.py).
    """

    def __init__(self, t_out, percentile_mode):
        import multiprocessing
        self._t_out = multiprocessing.Value("d", float(t_out))
        self._percentile = multiprocessing.Value("d", float(percentile_mode))

    def get(self):
        return self._t_out.value

    def get_percentile(self):
        return self._percentile.value

    def set(self, t_out, percentile_mode=None):
        with self._t_out.get_lock():
            self._t_out.value = float(t_out)
        if percentile_mode is not None:
            with self._percentile.get_lock():
                self._percentile.value = float(percentile_mode)


class ThresholdController(threading.Thread):
    """Background profiler that rewrites t_out during training.

    Observations arrive from the producer processes on `stats_q` as
    ("fast", idx, latency), ("migrate", idx, detect_s) and
    ("resume", idx, bg_s).

    On censoring: a migrated sample's true cost is unknown, because it was cut
    off at t_out. Feeding that cut-off value back in makes the threshold a
    fixed point of itself -- every censored observation is <= t_out, so once
    the migration rate reaches (100 - P)%, the recomputed percentile IS t_out
    and the update does nothing. The default "reconstructed" mode instead pairs
    each migration with its background resume time and uses detect_s + bg_s,
    which over-counts the re-executed stage and therefore errs high. See
    experiments/vuln-matrix/hmp/adaptive.py for the measured comparison.
    """

    def __init__(self, shared, stats_q, cfg, profile_dir=""):
        super(ThresholdController, self).__init__(daemon=True)
        import collections
        self.shared = shared
        self.stats_q = stats_q
        self.cfg = cfg
        self.profile_dir = profile_dir
        self.window = collections.deque(maxlen=int(cfg.adapt_window))
        self.flags = collections.deque(maxlen=int(cfg.adapt_window))
        self._pending = {}
        self._stop = threading.Event()
        self._new = 0
        self._mode = float(cfg.calib_percentile)
        self._mode_since = time.time()
        self.updates = 0
        self.escalations = 0
        self.deescalations = 0
        self.history = [shared.get()]
        self._fh = None

    def _trace(self, row):
        if not self.profile_dir:
            return
        try:
            os.makedirs(self.profile_dir, exist_ok=True)
            if self._fh is None:
                path = os.path.join(self.profile_dir, "threshold_trace.csv")
                new = not os.path.exists(path)
                self._fh = open(path, "a", newline="")
                self._w = csv.writer(self._fh)
                if new:
                    self._w.writerow(["wall", "update", "n_obs", "migration_rate",
                                      "percentile", "t_out_before", "t_out_after",
                                      "mode_change", "reason"])
            self._w.writerow(row)
            self._fh.flush()
        except OSError:
            self.profile_dir = ""

    def _ingest(self, item):
        kind = item[0]
        if kind == "fast":
            self.window.append(float(item[2]))
            self.flags.append(0)
            self._new += 1
        elif kind == "migrate":
            if self.cfg.adapt_censoring == "censored":
                self.window.append(float(item[2]))
                self.flags.append(1)
                self._new += 1
            else:
                self._pending[item[1]] = float(item[2])
        elif kind == "resume":
            detect = self._pending.pop(item[1], None)
            if detect is not None:
                self.window.append(detect + float(item[2]))
                self.flags.append(1)
                self._new += 1

    def _update(self, force=False):
        cfg = self.cfg
        if not force and (len(self.window) < cfg.adapt_min_samples
                          or self._new < cfg.adapt_min_samples):
            return
        self._new = 0
        ordered = sorted(self.window)
        rate = (sum(self.flags) / float(len(self.flags))) if self.flags else 0.0
        now = time.time()
        dwell = (now - self._mode_since) >= cfg.adapt_cooldown_s
        change, reason = "", "periodic"
        if self._mode == cfg.calib_percentile and rate > cfg.adapt_escalate_rate \
                and dwell:
            self._mode = float(cfg.adapt_fallback_percentile)
            self._mode_since, change = now, "escalate"
            self.escalations += 1
            reason = "migration_rate %.3f > %.3f" % (rate, cfg.adapt_escalate_rate)
        elif self._mode == cfg.adapt_fallback_percentile \
                and rate < cfg.adapt_deescalate_rate and dwell:
            self._mode = float(cfg.calib_percentile)
            self._mode_since, change = now, "deescalate"
            self.deescalations += 1
            reason = "migration_rate %.3f < %.3f" % (rate, cfg.adapt_deescalate_rate)

        before = self.shared.get()
        after = _percentile_of(ordered, self._mode) * cfg.calib_margin
        ratio = cfg.adapt_max_step_ratio
        if ratio and ratio > 1.0 and before > 0:
            after = min(max(after, before / ratio), before * ratio)
        after = min(max(after, cfg.t_out_min), cfg.t_out_max)
        self.shared.set(after, self._mode)
        self.updates += 1
        self.history.append(after)
        self._trace(["%.6f" % now, self.updates, len(self.window),
                     "%.4f" % rate, self._mode, "%.6f" % before,
                     "%.6f" % after, change, reason])
        if change:
            print("[minato/adapt] %s to P%.0f (%s); t_out %.4fs -> %.4fs"
                  % (change, self._mode, reason, before, after))

    def run(self):
        last = time.time()
        while not self._stop.is_set():
            try:
                item = self.stats_q.get(timeout=0.2)
                if item is not None:
                    self._ingest(item)
            except Exception:
                pass
            if time.time() - last >= self.cfg.adapt_interval_s:
                self._update()
                last = time.time()
        while True:
            try:
                item = self.stats_q.get_nowait()
            except Exception:
                break
            if item is not None:
                self._ingest(item)
        self._update(force=True)
        if self._fh is not None:
            self._fh.close()

    def stop(self):
        self._stop.set()

    def summary(self):
        hist = [h for h in self.history if h is not None]
        return {"adapt_updates": self.updates,
                "adapt_escalations": self.escalations,
                "adapt_deescalations": self.deescalations,
                "adapt_t_out_final": self.shared.get(),
                "adapt_t_out_min": min(hist) if hist else None,
                "adapt_t_out_max": max(hist) if hist else None,
                "adapt_percentile_final": self.shared.get_percentile()}


def _percentile_of(sorted_values, q):
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 100:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] * (1 - (pos - lo)) + sorted_values[hi] * (pos - lo)
