"""Baseline loader, downstream consumer, system monitor, metric aggregation."""

import csv
import glob
import json
import os
import statistics
import threading
import time

from hmp.minato import EventLog, _stage_json, run_stages


# ---------------------------------------------------------------------------
# plain PyTorch DataLoader baseline
# ---------------------------------------------------------------------------

class BaselineDataset(object):
    """torch Dataset over a run manifest, using the identical stage list.

    Constructed lazily inside each DataLoader worker so that the stage objects
    (and any C-level handles they hold) are never pickled across the fork.
    """

    def __init__(self, rows, cfg):
        self.rows = rows
        self.cfg = cfg
        self._stages = None
        self._log = None

    def __len__(self):
        return len(self.rows)

    def _ensure(self):
        if self._stages is None:
            from hmp import pipelines
            self._stages = pipelines.build_stages(
                self.cfg["pipeline"], self.cfg["pipeline_kwargs"],
                seed=self.cfg["seed"])
            self._log = EventLog(self.cfg["events_dir"], "baseline")
        return self._stages

    def __getitem__(self, i):
        stages = self._ensure()
        row = self.rows[i]
        timings = []
        t0 = time.perf_counter()
        try:
            payload = run_stages(stages, {"row": row}, 0, None, None, timings)
            out = payload["out"]
            dur = time.perf_counter() - t0
            self._log.write(event="baseline", order=row["order"],
                            sample_id=row["sample_id"],
                            truth_label=row["truth_label"],
                            is_sentinel=int(bool(row["is_sentinel"])),
                            sentinel_role=row.get("sentinel_role", ""),
                            duration_s="%.6f" % dur, resume_stage=len(stages),
                            n_stages=len(stages),
                            stage_times_json=_stage_json(timings))
            return out, row["order"], 1 if row["truth_label"] == "slow" else 0, dur
        except Exception as exc:
            self._log.write(event="baseline_error", order=row["order"],
                            sample_id=row["sample_id"],
                            truth_label=row["truth_label"],
                            stage_times_json=json.dumps([["error", repr(exc)]]))
            import torch
            return torch.zeros(1), row["order"], 0, 0.0


def baseline_loader(rows, cfg):
    import torch
    from torch.utils.data import DataLoader

    def collate(items):
        tensors = [it[0] for it in items]
        try:
            batch = torch.stack(tensors)
            if batch.dim() == 5:
                # SimCLR: (B, views, C, H, W) -> (B*views, C, H, W), the shape
                # a contrastive loss consumes.
                batch = batch.flatten(0, 1)
        except Exception:
            batch = tensors
        return (batch, [it[1] for it in items], sum(it[2] for it in items),
                [it[3] for it in items])

    return DataLoader(BaselineDataset(rows, cfg),
                      batch_size=cfg["batch_size"],
                      num_workers=cfg["baseline_workers"],
                      shuffle=False, drop_last=False,
                      pin_memory=cfg.get("pin_memory", False),
                      prefetch_factor=cfg.get("prefetch_factor", 2)
                      if cfg["baseline_workers"] > 0 else None,
                      persistent_workers=False,
                      collate_fn=collate)


# ---------------------------------------------------------------------------
# downstream consumer
# ---------------------------------------------------------------------------

class Consumer(object):
    """Emulates the training step that the loader has to keep fed.

    Measuring loader throughput with no consumer at all is not meaningful for
    this comparison: MinatoLoader's whole claim is about overlapping slow-sample
    work with GPU compute, and with no compute there is nothing to overlap with.
    Default is a fixed-duration synthetic step; `model:<name>` runs a real
    forward/backward on the GPU when one is present.
    """

    def __init__(self, spec, device="cpu"):
        self.spec = spec or "none"
        self.device = device
        self.kind, _, arg = self.spec.partition(":")
        self.arg = arg
        self.model = None
        self.optimizer = None
        if self.kind == "model":
            self._build_model(arg or "resnet18")

    def _build_model(self, name):
        import torch
        import torchvision
        self.model = getattr(torchvision.models, name)(weights=None)
        self.model.to(self.device).train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self._torch = torch

    def step(self, batch):
        if self.kind == "none":
            return 0.0
        t0 = time.perf_counter()
        if self.kind == "sleep":
            time.sleep(float(self.arg or 0.03))
        elif self.kind == "model":
            torch = self._torch
            x = batch if hasattr(batch, "to") else None
            if x is None:
                return time.perf_counter() - t0
            x = x.to(self.device, non_blocking=True)
            if x.dim() == 3:
                x = x.unsqueeze(1).repeat(1, 3, 1, 1)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = out.float().pow(2).mean()
            loss.backward()
            self.optimizer.step()
            if self.device != "cpu":
                torch.cuda.synchronize()
        return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# system monitor
# ---------------------------------------------------------------------------

class SystemMonitor(threading.Thread):
    """Samples CPU / memory, and GPU when NVML is importable. No-op fallback."""

    def __init__(self, out_path, interval=0.5, per_core=True):
        super(SystemMonitor, self).__init__(daemon=True)
        self.out_path = out_path
        self.interval = interval
        self.per_core = per_core
        self._stop = threading.Event()
        self.gpu = None
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu = pynvml
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.gpu = None

    def run(self):
        if self.psutil is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)) or ".",
                    exist_ok=True)
        with open(self.out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            header = ["t", "cpu_percent", "mem_used_mb", "gpu_util", "gpu_mem_mb"]
            if self.per_core:
                header.append("per_core_json")
            writer.writerow(header)
            while not self._stop.is_set():
                cpu = self.psutil.cpu_percent(interval=None)
                per_core = self.psutil.cpu_percent(interval=None, percpu=True) \
                    if self.per_core else []
                mem = self.psutil.virtual_memory().used / (1024 ** 2)
                gpu_util = gpu_mem = ""
                if self.gpu is not None:
                    try:
                        u = self.gpu.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        m = self.gpu.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        gpu_util = u.gpu
                        gpu_mem = m.used / (1024 ** 2)
                    except Exception:
                        pass
                row = ["%.3f" % time.time(), cpu, "%.1f" % mem, gpu_util, gpu_mem]
                if self.per_core:
                    row.append(json.dumps(per_core))
                writer.writerow(row)
                fh.flush()
                self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = [
    "run_id", "dataset", "pipeline", "regime", "slow_fraction", "loader",
    "n_foreground", "n_background", "baseline_workers", "batch_size",
    "n_samples", "manifest_fingerprint", "seed", "consumer",
    "t_out_s", "t_out_percentile", "warmup_n", "warmup_p50_s", "warmup_p95_s",
    "warmup_slow_below_t_out_frac",
    "adapt", "adapt_censoring", "adapt_updates", "adapt_escalations",
    "adapt_deescalations", "adapt_t_out_final", "adapt_t_out_min",
    "adapt_t_out_max", "adapt_percentile_final", "adapt_seconds_at_fallback",
    "t_out_at_migration_p50", "t_out_at_migration_max",
    "epoch_wall_s", "throughput_samples_s", "consumer_wall_s",
    "sample_p50_s", "sample_p95_s", "sample_p99_s",
    "batch_mean_s", "batch_p50_s", "batch_p95_s", "batch_p99_s",
    "n_fast_route", "n_slow_route", "n_errors",
    "migrations", "migrations_at_stage_0", "mean_resume_stage",
    "detect_tax_total_s", "detect_tax_per_sample_s",
    "wasted_cpu_resume_s", "wasted_cpu_full_restart_s",
    "tp_slow_migrated", "fn_slow_kept_fast", "fp_fast_migrated",
    "tn_fast_kept_fast", "misclassification_rate", "slow_recall",
    "early_batch_slow_frac", "late_batch_slow_frac", "batch_composition_skew",
    "sentinel_fast_p50_s", "sentinel_fast_p95_s",
    "sentinel_slow_p50_s", "sentinel_slow_p95_s",
    "cpu_percent_mean", "gpu_util_mean", "mem_used_mb_max",
]


def _q(values, p):
    if not values:
        return ""
    s = sorted(values)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    pos = (len(s) - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (pos - lo)) + s[hi] * (pos - lo)


def read_events(run_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "events", "events_*.csv"))):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                for k in ("duration_s", "queue_wait_s", "detect_s", "wasted_s",
                          "preserved_s", "t_out_s"):
                    r[k] = float(r[k]) if r.get(k) not in (None, "") else 0.0
                for k in ("order", "resume_stage", "n_stages", "is_sentinel"):
                    try:
                        r[k] = int(r[k])
                    except (TypeError, ValueError):
                        r[k] = -1
                rows.append(r)
    return rows


def read_batches(run_dir):
    path = os.path.join(run_dir, "batches.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            for k in ("batch_index", "size", "n_truth_slow", "n_route_slow",
                      "n_error"):
                r[k] = int(float(r[k] or 0))
            for k in ("latency_s", "mean_item_wait_s", "consumer_s"):
                r[k] = float(r[k] or 0.0)
            out.append(r)
    return out


def summarise(run_dir, meta):
    """Fold one run directory into a single summary row."""
    events = read_events(run_dir)
    batches = read_batches(run_dir)
    row = {k: "" for k in SUMMARY_FIELDS}
    row.update({k: v for k, v in meta.items() if k in SUMMARY_FIELDS})

    completed = [e for e in events
                 if e["event"] in ("fast", "resume", "resume_restart", "baseline")]
    migrations = [e for e in events if e["event"] == "migrate"]
    errors = [e for e in events if e["event"].endswith("error")]

    durations = [e["duration_s"] for e in completed if e["duration_s"] > 0]
    row["sample_p50_s"] = _q(durations, 50)
    row["sample_p95_s"] = _q(durations, 95)
    row["sample_p99_s"] = _q(durations, 99)
    row["n_errors"] = len(errors)

    if batches:
        lat = [b["latency_s"] for b in batches]
        row["batch_mean_s"] = statistics.fmean(lat)
        row["batch_p50_s"] = _q(lat, 50)
        row["batch_p95_s"] = _q(lat, 95)
        row["batch_p99_s"] = _q(lat, 99)
        half = max(1, len(batches) // 4)
        early = batches[:half]
        late = batches[-half:]
        ef = sum(b["n_truth_slow"] for b in early) / max(1, sum(b["size"] for b in early))
        lf = sum(b["n_truth_slow"] for b in late) / max(1, sum(b["size"] for b in late))
        row["early_batch_slow_frac"] = ef
        row["late_batch_slow_frac"] = lf
        # Positive => early batches are depleted of slow samples relative to
        # late ones, i.e. the epoch's data distribution is time-skewed.
        row["batch_composition_skew"] = lf - ef

    # --- migration / classification -------------------------------------
    row["migrations"] = len(migrations)
    row["migrations_at_stage_0"] = sum(1 for e in migrations if e["resume_stage"] == 0)
    resumes = [e for e in events if e["event"].startswith("resume")]
    if resumes:
        row["mean_resume_stage"] = statistics.fmean(
            [e["resume_stage"] for e in resumes])
    detect = sum(e["detect_s"] for e in migrations)
    row["detect_tax_total_s"] = detect
    n_total = meta.get("n_samples") or max(1, len(completed))
    row["detect_tax_per_sample_s"] = detect / float(n_total)
    row["wasted_cpu_resume_s"] = sum(e["wasted_s"] for e in migrations)
    tv = [e["t_out_s"] for e in migrations if e.get("t_out_s")]
    if tv:
        row["t_out_at_migration_p50"] = _q(tv, 50)
        row["t_out_at_migration_max"] = _q(tv, 100)

    stage_median = meta.get("stage_median_s") or {}
    full_cost = sum(stage_median.values()) if stage_median else 0.0
    row["wasted_cpu_full_restart_s"] = detect + len(migrations) * full_cost

    migrated_ids = set(e["sample_id"] for e in migrations)
    truth = {}
    for e in events:
        if e.get("sample_id") and e.get("truth_label") in ("fast", "slow"):
            truth.setdefault(e["sample_id"], e["truth_label"])
    tp = sum(1 for s, t in truth.items() if t == "slow" and s in migrated_ids)
    fn = sum(1 for s, t in truth.items() if t == "slow" and s not in migrated_ids)
    fp = sum(1 for s, t in truth.items() if t == "fast" and s in migrated_ids)
    tn = sum(1 for s, t in truth.items() if t == "fast" and s not in migrated_ids)
    row["tp_slow_migrated"] = tp
    row["fn_slow_kept_fast"] = fn
    row["fp_fast_migrated"] = fp
    row["tn_fast_kept_fast"] = tn
    denom = tp + fn + fp + tn
    row["misclassification_rate"] = (fp + fn) / float(denom) if denom else ""
    row["slow_recall"] = tp / float(tp + fn) if (tp + fn) else ""

    routes = [e["event"] for e in completed]
    row["n_fast_route"] = sum(1 for r in routes if r in ("fast", "baseline"))
    row["n_slow_route"] = sum(1 for r in routes if r.startswith("resume"))

    # --- sentinels --------------------------------------------------------
    for role in ("fast", "slow"):
        vals = [e["duration_s"] for e in completed
                if e.get("sentinel_role") == role and e["duration_s"] > 0]
        row["sentinel_%s_p50_s" % role] = _q(vals, 50)
        row["sentinel_%s_p95_s" % role] = _q(vals, 95)

    # --- system -----------------------------------------------------------
    sys_path = os.path.join(run_dir, "system.csv")
    if os.path.exists(sys_path):
        cpu, gpu, mem = [], [], []
        with open(sys_path, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    cpu.append(float(r["cpu_percent"]))
                except (ValueError, KeyError):
                    pass
                try:
                    gpu.append(float(r["gpu_util"]))
                except (ValueError, KeyError, TypeError):
                    pass
                try:
                    mem.append(float(r["mem_used_mb"]))
                except (ValueError, KeyError):
                    pass
        row["cpu_percent_mean"] = statistics.fmean(cpu) if cpu else ""
        row["gpu_util_mean"] = statistics.fmean(gpu) if gpu else ""
        row["mem_used_mb_max"] = max(mem) if mem else ""

    return row


def append_summary(summary_csv, row):
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv)) or ".",
                exist_ok=True)
    new = not os.path.exists(summary_csv)
    with open(summary_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


def plan_cores(n_fg, n_bg, front_threads=1, reserve=2):
    """Assign disjoint core sets. Linux only; returns {} where unsupported."""
    if not hasattr(os, "sched_getaffinity"):
        return {}
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < n_fg + n_bg + reserve:
        return {}
    pool = cores[reserve:]
    plan = {"fg": {}, "bg": {}, "main": cores[:reserve]}
    i = 0
    for w in range(n_fg):
        plan["fg"][w] = [pool[(i + k) % len(pool)] for k in range(front_threads)]
        i += front_threads
    for w in range(n_bg):
        plan["bg"][w] = [pool[i % len(pool)]]
        i += 1
    return plan
