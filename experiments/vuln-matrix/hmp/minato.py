"""MinatoLoader mechanism, generalised over arbitrary sample pipelines.

This is a port of the corrected loader (calibrated t_out + resume-from-index)
out of Minato/data_loading/ and into a dataset-agnostic harness, because the
in-tree loader is welded to the 3D .npy volume layout and cannot run OpenImages
or LibriSpeech. The semantics are the same:

  * the pipeline is an indexed list of stages; position == resume index
  * a foreground worker runs stages in a thread, checkpointing a private copy
    after each one and checking an abort flag at every boundary
  * on timeout it ships (order, resume_stage, payload) to the background
    workers, which resume at resume_stage
  * t_out comes from a warm-up profiling pass over the run's own mixed dataset

Differences from the in-tree loader, all deliberate and all recorded in the run
metadata so they cannot be quietly forgotten:

  * t_out is calibrated ONCE in the parent and handed to the workers, rather
    than per-dataset-object. The design brief fixes t_out as an experimental
    variable, so it must not drift between workers.
  * adaptive threshold adjustment and adaptive worker scaling are NOT
    implemented. F and B are fixed by the caller.
  * every sample carries its ground-truth pool label through the pipeline, so
    the runtime classification can be scored against it.
"""

import csv
import json
import os
import queue as _queue
import threading
import time

try:
    import torch.multiprocessing as mp
except Exception:  # torch-free selftest path
    import multiprocessing as mp

STOP = None


# ---------------------------------------------------------------------------
# indexed pipeline execution
# ---------------------------------------------------------------------------

class StageAborted(Exception):
    def __init__(self, stage):
        super(StageAborted, self).__init__("aborted after stage %d" % stage)
        self.stage = stage


def clone_payload(payload):
    """Private copy of a sample. Stages may write into their input in place."""
    out = {}
    for key, value in payload.items():
        clone = getattr(value, "clone", None)          # torch.Tensor
        if callable(clone):
            out[key] = clone()
            continue
        copy = getattr(value, "copy", None)            # ndarray, PIL, dict
        if callable(copy):
            try:
                out[key] = copy()
                continue
            except TypeError:
                pass
        out[key] = value                                # immutable / path str
    return out


def run_stages(stages, payload, start_index=0, checkpoint=None,
               abort_event=None, timings=None):
    """Execute stages[start_index:]. See the module docstring."""
    for i in range(start_index, len(stages)):
        if abort_event is not None and abort_event.is_set():
            raise StageAborted(i)
        name, fn = stages[i]
        t0 = time.perf_counter()
        payload = fn(payload)
        if timings is not None:
            timings.append((name, time.perf_counter() - t0))
        if checkpoint is not None:
            if abort_event is not None and abort_event.is_set():
                raise StageAborted(i)
            checkpoint(i + 1, clone_payload(payload))
    return payload


def _percentile(sorted_values, q):
    if q <= 0:
        return sorted_values[0]
    if q >= 100:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


# ---------------------------------------------------------------------------
# warm-up calibration
# ---------------------------------------------------------------------------

def calibrate_t_out(stages, rows, warmup_samples, percentile, seed,
                    margin=1.0, discard=3, log=print):
    """Warm-up profiling pass -> t_out.

    Runs over a random subset of the run's OWN mixed manifest, which is the
    behaviour that makes vulnerability #2 visible: at slow_fraction > 1 - P/100
    the percentile cut lands inside the slow population, so part of the slow
    pool is classified fast by construction.

    Measured through the same thread-handoff path the foreground workers use.
    Measuring synchronously here and comparing against a threaded run at epoch
    time biases t_out low by the handoff cost, which on a tight distribution is
    enough to migrate every sample.
    """
    import concurrent.futures
    import random

    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    budget = min(warmup_samples + discard, len(order))

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                     thread_name_prefix="warmup")
    totals, per_stage, per_label = [], {}, {"fast": [], "slow": []}
    t_start = time.perf_counter()
    try:
        for n, idx in enumerate(order[:budget]):
            row = rows[idx]
            timings = []
            started = threading.Event()

            def work(row=row, timings=timings, started=started):
                started.set()
                return run_stages(stages, {"row": row}, 0, None, None, timings)

            future = executor.submit(work)
            started.wait(timeout=60)
            t0 = time.perf_counter()
            try:
                future.result()
            except Exception as exc:
                log("[calib] sample %s failed: %r" % (row["sample_id"], exc))
                continue
            total = time.perf_counter() - t0
            if n < discard:          # cold page cache / allocator warm-up
                continue
            totals.append(total)
            per_label.setdefault(row.get("truth_label", "?"), []).append(total)
            for name, dt in timings:
                per_stage.setdefault(name, []).append(dt)
    finally:
        executor.shutdown(wait=True)

    if len(totals) < 2:
        raise RuntimeError("warm-up produced %d usable timings" % len(totals))

    ordered = sorted(totals)
    raw = _percentile(ordered, percentile)
    t_out = raw * margin

    result = {
        "t_out": t_out,
        "t_out_raw": raw,
        "percentile": percentile,
        "margin": margin,
        "n_warmup": len(totals),
        "warmup_wall_s": time.perf_counter() - t_start,
        "stages": [name for name, _ in stages],
        "latency": {("p%g" % q): _percentile(ordered, q)
                    for q in (0, 25, 50, 75, 90, 95, 99, 100)},
        "stage_median_s": {k: _percentile(sorted(v), 50)
                           for k, v in per_stage.items()},
        "warmup_by_truth_label": {
            k: {"n": len(v),
                "p50": _percentile(sorted(v), 50) if v else None,
                "p95": _percentile(sorted(v), 95) if v else None}
            for k, v in per_label.items() if v},
    }
    # How much of the slow pool sits below the cut, i.e. is fast by definition.
    slow_warm = sorted(per_label.get("slow", []))
    if slow_warm:
        below = sum(1 for v in slow_warm if v <= t_out)
        result["warmup_slow_below_t_out_frac"] = below / float(len(slow_warm))
    log("[calib] t_out = %.4fs (P%.0f x %.2f over %d warm-up samples; "
        "p50=%.4f p95=%.4f)"
        % (t_out, percentile, margin, len(totals),
           result["latency"]["p50"], result["latency"]["p95"]))
    return result


# ---------------------------------------------------------------------------
# event recording
# ---------------------------------------------------------------------------

EVENT_FIELDS = [
    "wall", "role", "pid", "event", "order", "sample_id", "truth_label",
    "is_sentinel", "sentinel_role", "duration_s", "queue_wait_s",
    "detect_s", "wasted_s", "preserved_s", "resume_stage", "n_stages",
    "t_out_s", "stage_times_json",
]


class EventLog(object):
    """One CSV per process. No cross-process locking, no IPC bottleneck."""

    def __init__(self, events_dir, role):
        self.dir = events_dir
        self.role = role
        self._pid = None
        self._fh = None
        self._w = None
        self._lock = threading.Lock()

    def _ensure(self):
        pid = os.getpid()
        if self._fh is not None and self._pid == pid:
            return True
        self._pid = pid
        os.makedirs(self.dir, exist_ok=True)
        path = os.path.join(self.dir, "events_%s_%d.csv" % (self.role, pid))
        new = not os.path.exists(path)
        self._fh = open(path, "a", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=EVENT_FIELDS)
        if new:
            self._w.writeheader()
        return True

    def write(self, **kw):
        with self._lock:
            self._ensure()
            row = {k: "" for k in EVENT_FIELDS}
            row.update({k: v for k, v in kw.items() if k in EVENT_FIELDS})
            row["wall"] = "%.6f" % time.time()
            row["role"] = self.role
            row["pid"] = os.getpid()
            self._w.writerow(row)
            self._fh.flush()

    def close(self):
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def _stage_json(timings):
    return json.dumps([[n, round(d, 6)] for n, d in (timings or [])])


def _pin(cores):
    if not cores:
        return
    setaffinity = getattr(os, "sched_setaffinity", None)
    if setaffinity is None:          # macOS: not available, documented as such
        return
    try:
        setaffinity(0, set(cores))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------

def foreground_worker(worker_id, cfg, index_q, fast_q, slow_q, ready_ev, cores,
                      threshold_handles=None, stats_q=None):
    """Pull indices, run the pipeline under t_out, migrate on timeout.

    ``t_out`` is re-read from shared memory once per sample, so a threshold
    rewritten by the controller takes effect on the next sample rather than at
    the next epoch. A stale read only means one sample was judged against the
    previous value.
    """
    import concurrent.futures
    from hmp import pipelines
    from hmp.adaptive import WorkerThresholdView

    _pin(cores)
    stages = pipelines.build_stages(cfg["pipeline"], cfg["pipeline_kwargs"],
                                    seed=cfg["seed"] + worker_id)
    n_stages = len(stages)
    threshold = WorkerThresholdView(threshold_handles) if threshold_handles else None
    log = EventLog(cfg["events_dir"], "fg")
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=cfg["front_threads"], thread_name_prefix="fg%d" % worker_id)
    ready_ev.set()

    try:
        while True:
            try:
                item = index_q.get(timeout=cfg["idle_timeout_s"])
            except Exception:
                break
            if item is STOP:
                break

            t_out = threshold.get() if threshold is not None else cfg["t_out"]
            state = {"stage": 0, "payload": None}
            lock = threading.Lock()
            abort = threading.Event()
            started = threading.Event()
            timings = []

            def checkpoint(stage, payload):
                with lock:
                    state["stage"] = stage
                    state["payload"] = payload

            def work():
                started.set()
                return run_stages(stages, {"row": item}, 0, checkpoint, abort,
                                  timings)

            submitted = time.perf_counter()
            future = executor.submit(work)
            # The deadline starts when the pool picks the task up, not when it
            # is queued; otherwise queueing behind an abandoned stage reads as
            # a slow sample.
            started.wait(timeout=cfg["idle_timeout_s"])
            qwait = time.perf_counter() - submitted
            t0 = time.perf_counter()

            common = dict(order=item["order"], sample_id=item["sample_id"],
                          truth_label=item["truth_label"],
                          is_sentinel=int(bool(item["is_sentinel"])),
                          sentinel_role=item.get("sentinel_role", ""),
                          n_stages=n_stages, queue_wait_s="%.6f" % qwait,
                          t_out_s="%.6f" % t_out)

            try:
                payload = future.result(timeout=t_out)
                dur = time.perf_counter() - t0
                fast_q.put({"order": item["order"], "sample_id": item["sample_id"],
                            "truth_label": item["truth_label"],
                            "is_sentinel": item["is_sentinel"],
                            "sentinel_role": item.get("sentinel_role", ""),
                            "route": "fast", "fg_s": dur, "bg_s": 0.0,
                            "resume_stage": n_stages,
                            "tensor": payload.get("out")})
                log.write(event="fast", duration_s="%.6f" % dur,
                          stage_times_json=_stage_json(timings), **common)
                if stats_q is not None:
                    try:
                        stats_q.put(("fast", item["order"], dur))
                    except Exception:
                        pass
            except concurrent.futures.TimeoutError:
                abort.set()
                future.add_done_callback(lambda f: f.exception())
                detect = time.perf_counter() - t0
                with lock:
                    stage, partial = state["stage"], state["payload"]
                preserved = sum(d for _, d in timings[:stage])
                slow_q.put((item, stage, partial))
                log.write(event="migrate", duration_s="%.6f" % detect,
                          detect_s="%.6f" % detect,
                          preserved_s="%.6f" % preserved,
                          wasted_s="%.6f" % max(0.0, detect - preserved),
                          resume_stage=stage,
                          stage_times_json=_stage_json(timings), **common)
                if stats_q is not None:
                    try:
                        stats_q.put(("migrate", item["order"], detect))
                    except Exception:
                        pass
            except Exception as exc:
                log.write(event="fg_error", duration_s="0",
                          stage_times_json=json.dumps([["error", repr(exc)]]),
                          **common)
                fast_q.put({"order": item["order"], "sample_id": item["sample_id"],
                            "truth_label": item["truth_label"],
                            "is_sentinel": item["is_sentinel"],
                            "sentinel_role": item.get("sentinel_role", ""),
                            "route": "error", "fg_s": 0.0, "bg_s": 0.0,
                            "resume_stage": -1, "tensor": None})
    finally:
        executor.shutdown(wait=False)
        log.close()


def background_worker(worker_id, cfg, slow_q, resumed_q, ready_ev, cores,
                      stats_q=None):
    """Resume migrated samples at their transformation index."""
    from hmp import pipelines

    _pin(cores)
    stages = pipelines.build_stages(cfg["pipeline"], cfg["pipeline_kwargs"],
                                    seed=cfg["seed"] + 1000 + worker_id)
    n_stages = len(stages)
    log = EventLog(cfg["events_dir"], "bg")
    ready_ev.set()

    try:
        while True:
            try:
                item = slow_q.get(timeout=cfg["idle_timeout_s"])
            except Exception:
                break
            if item is STOP:
                break
            row, resume_stage, partial = item
            resume_stage = max(0, min(int(resume_stage), n_stages))
            restarted = partial is None
            if restarted:
                payload, resume_stage = {"row": row}, 0
            else:
                payload = partial

            timings = []
            t0 = time.perf_counter()
            try:
                payload = run_stages(stages, payload, resume_stage, None, None,
                                     timings)
                dur = time.perf_counter() - t0
                resumed_q.put({"order": row["order"], "sample_id": row["sample_id"],
                               "truth_label": row["truth_label"],
                               "is_sentinel": row["is_sentinel"],
                               "sentinel_role": row.get("sentinel_role", ""),
                               "route": "slow", "fg_s": 0.0, "bg_s": dur,
                               "resume_stage": resume_stage,
                               "tensor": payload.get("out")})
                if stats_q is not None:
                    try:
                        stats_q.put(("resume", row["order"], dur))
                    except Exception:
                        pass
                log.write(event="resume" if not restarted else "resume_restart",
                          order=row["order"], sample_id=row["sample_id"],
                          truth_label=row["truth_label"],
                          is_sentinel=int(bool(row["is_sentinel"])),
                          sentinel_role=row.get("sentinel_role", ""),
                          duration_s="%.6f" % dur, resume_stage=resume_stage,
                          n_stages=n_stages, stage_times_json=_stage_json(timings))
            except Exception as exc:
                log.write(event="bg_error", order=row["order"],
                          sample_id=row["sample_id"],
                          truth_label=row["truth_label"],
                          resume_stage=resume_stage, n_stages=n_stages,
                          stage_times_json=json.dumps([["error", repr(exc)]]))
                resumed_q.put({"order": row["order"], "sample_id": row["sample_id"],
                               "truth_label": row["truth_label"],
                               "is_sentinel": row["is_sentinel"],
                               "sentinel_role": row.get("sentinel_role", ""),
                               "route": "error", "fg_s": 0.0, "bg_s": 0.0,
                               "resume_stage": -1, "tensor": None})
    finally:
        log.close()


# ---------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------

class MinatoHarness(object):
    """Fixed F/B split, fixed t_out, fast-queue-preferential batching."""

    def __init__(self, rows, cfg, trace_path=None):
        from hmp.adaptive import SharedThreshold, ThresholdController

        self.rows = rows
        self.cfg = cfg
        self.n = len(rows)
        ctx = mp
        self.index_q = ctx.Queue()
        self.fast_q = ctx.Queue(maxsize=cfg["fast_queue_size"])
        self.slow_q = ctx.Queue()
        self.resumed_q = ctx.Queue()
        self.procs = []
        self.queue_trace = []
        self._stop_trace = threading.Event()

        # t_out lives in shared memory whether or not adaptation is enabled,
        # so the fast and adaptive paths run identical worker code.
        self.threshold = SharedThreshold(cfg["t_out"],
                                         cfg.get("adapt_percentile", 75.0))
        self.adapt = bool(cfg.get("adapt"))
        self.stats_q = ctx.Queue() if self.adapt else None
        self.controller = None
        if self.adapt:
            self.controller = ThresholdController(
                self.threshold, self.stats_q, cfg, trace_path=trace_path,
                log=print)

    def start(self):
        cfg = self.cfg
        for row in self.rows:
            self.index_q.put(row)
        for _ in range(cfg["n_foreground"]):
            self.index_q.put(STOP)

        cores = cfg.get("core_plan") or {}
        for i in range(cfg["n_foreground"]):
            ev = mp.Event()
            p = mp.Process(target=foreground_worker,
                           args=(i, cfg, self.index_q, self.fast_q, self.slow_q,
                                 ev, cores.get("fg", {}).get(i),
                                 self.threshold.handles(), self.stats_q),
                           daemon=True)
            p.start()
            self.procs.append((p, ev))
        for i in range(cfg["n_background"]):
            ev = mp.Event()
            p = mp.Process(target=background_worker,
                           args=(i, cfg, self.slow_q, self.resumed_q, ev,
                                 cores.get("bg", {}).get(i), self.stats_q),
                           daemon=True)
            p.start()
            self.procs.append((p, ev))
        for p, ev in self.procs:
            ev.wait(timeout=120)

        t = threading.Thread(target=self._trace_queues, daemon=True)
        t.start()
        self._trace_thread = t
        if self.controller is not None:
            self.controller.start()

    def _trace_queues(self):
        while not self._stop_trace.is_set():
            try:
                self.queue_trace.append({
                    "t": time.time(),
                    "fast": self.fast_q.qsize(),
                    "slow": self.slow_q.qsize(),
                    "resumed": self.resumed_q.qsize(),
                    "t_out": self.threshold.get(),
                    "percentile": self.threshold.get_percentile(),
                })
            except NotImplementedError:      # qsize() unavailable on macOS
                self.queue_trace.append({"t": time.time(), "fast": -1,
                                         "slow": -1, "resumed": -1,
                                         "t_out": self.threshold.get(),
                                         "percentile":
                                             self.threshold.get_percentile()})
            self._stop_trace.wait(self.cfg["queue_trace_interval_s"])

    def iter_batches(self):
        """Yield (batch, batch_meta). Fast queue first, slow queue as top-up.

        The top-up rule mirrors the artifact: drain the resumed queue only once
        it has backed up past `slow_drain_threshold`, which is what biases
        early batches toward cheap samples.
        """
        cfg = self.cfg
        bs = cfg["batch_size"]
        received = 0
        batch_index = 0
        max_batches = cfg.get("max_batches") or 0
        deadline = time.time() + cfg["epoch_timeout_s"]

        while received < self.n and time.time() < deadline:
            if max_batches and batch_index >= max_batches:
                break
            batch, t_batch0 = [], time.perf_counter()
            waits = []
            while len(batch) < bs and received + len(batch) < self.n:
                if time.time() > deadline:
                    break
                got = None
                w0 = time.perf_counter()
                try:
                    got = self.fast_q.get(timeout=cfg["queue_poll_s"])
                except Exception:
                    got = None
                if got is None:
                    try:
                        if self.resumed_q.qsize() > cfg["slow_drain_threshold"] \
                                or self._fast_exhausted():
                            got = self.resumed_q.get(timeout=cfg["queue_poll_s"])
                    except Exception:
                        got = None
                else:
                    try:
                        if self.resumed_q.qsize() > cfg["slow_drain_threshold"] \
                                and len(batch) + 1 < bs:
                            extra = self.resumed_q.get_nowait()
                            waits.append(time.perf_counter() - w0)
                            batch.append(extra)
                    except Exception:
                        pass
                if got is not None:
                    waits.append(time.perf_counter() - w0)
                    batch.append(got)
            if not batch:
                continue
            received += len(batch)
            meta = {
                "batch_index": batch_index,
                "size": len(batch),
                "latency_s": time.perf_counter() - t_batch0,
                "mean_item_wait_s": sum(waits) / len(waits) if waits else 0.0,
                "n_truth_slow": sum(1 for b in batch if b["truth_label"] == "slow"),
                "n_route_slow": sum(1 for b in batch if b["route"] == "slow"),
                "n_error": sum(1 for b in batch if b["route"] == "error"),
                "received_total": received,
            }
            batch_index += 1
            yield batch, meta

        self.received = received

    def _fast_exhausted(self):
        try:
            return self.fast_q.qsize() == 0 and self.index_q.qsize() == 0
        except NotImplementedError:
            return True

    def adapt_summary(self):
        return self.controller.summary() if self.controller is not None else {}

    def shutdown(self):
        self._stop_trace.set()
        if self.controller is not None:
            self.controller.stop()
            self.controller.join(timeout=15)
        for _ in range(self.cfg["n_background"]):
            try:
                self.slow_q.put(STOP)
            except Exception:
                pass
        for p, _ in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
