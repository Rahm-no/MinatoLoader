
import random
import numpy as np
import scipy.ndimage
from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch.multiprocessing as mp

try:  # optional: lets the loader be imported without the mlperf/dllogger stack
    from runtime.logging import mllog_event
except Exception:  # pragma: no cover
    def mllog_event(*args, **kwargs):
        pass

try:
    from data_loading.minato_profiling import (
        MigrationRecorder, MinatoConfig, SharedThreshold, StageAborted,
        ThresholdController, TimeoutCalibrator, build_stages, clone_data,
        run_stages, stage_names,
    )
except ImportError:  # running from inside data_loading/
    from minato_profiling import (
        MigrationRecorder, MinatoConfig, SharedThreshold, StageAborted,
        ThresholdController, TimeoutCalibrator, build_stages, clone_data,
        run_stages, stage_names,
    )

# scipy >= 1.10 deprecated `scipy.ndimage.measurements`; it is scheduled for
# removal in 2.0 and already raises under -W error on recent laptops.
_ndimage_label = getattr(scipy.ndimage, "label", None)
if _ndimage_label is None:  # pragma: no cover
    _ndimage_label = scipy.ndimage.measurements.label


def get_train_transforms():
    rand_flip = RandFlip()
    cast = Cast(types=(np.float32, np.uint8))
    heavy_blur = HeavyGaussianBlur(sigma_range=(1, 5))
    elastic_deform = ElasticDeformation(alpha=1000, sigma=30)
    rand_scale = RandomBrightnessAugmentation(factor=0.3, prob=0.1)
    rand_noise = GaussianNoise(mean=0.0, std=0.1, prob=0.1)
    
    train_transforms = transforms.Compose([rand_flip, cast, rand_scale, rand_noise])
    return train_transforms




class HeavyGaussianBlur:
    def __init__(self, sigma_range=(1, 5)):
        self.sigma_range = sigma_range

    def __call__(self, data):
        sigma = np.random.uniform(*self.sigma_range)
        image = data["image"]
        for c in range(image.shape[0]):
            image[c] = scipy.ndimage.gaussian_filter(image[c], sigma=sigma)
        data["image"] = image
        return data


class ElasticDeformation:
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, data):
        image = data["image"]
        shape = image.shape[1:]

        # Generate random displacement fields
        dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dz = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha

        # Create meshgrid
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = (x + dx).reshape(-1), (y + dy).reshape(-1), (z + dz).reshape(-1)

        # Apply deformation
        for c in range(image.shape[0]):
            image[c] = scipy.ndimage.map_coordinates(image[c], indices, order=1, mode='reflect').reshape(shape)

        data["image"] = image
        return data



class RandBalancedCrop:
    def __init__(self, patch_size, oversampling):
        self.patch_size = patch_size
        self.oversampling = oversampling

    def __call__(self, data):
        image, label = data["image"], data["label"]
        image_new, label_new, cords_new = self.rand_foreg_cropd(image, label)
    
        image1, label1, cords1 = self._rand_crop(image, label)
        data.update({"image": image_new, "label": label_new})
        return data

    @staticmethod
    def randrange(max_range):
        return 0 if max_range == 0 else random.randrange(max_range)

    def get_cords(self, cord, idx):
        return cord[idx], cord[idx] + self.patch_size[idx]

    def _rand_crop(self, image, label):
        ranges = [s - p for s, p in zip(image.shape[1:], self.patch_size)]
        cord = [self.randrange(x) for x in ranges]
        low_x, high_x = self.get_cords(cord, 0)
        low_y, high_y = self.get_cords(cord, 1)
        low_z, high_z = self.get_cords(cord, 2)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]

    def rand_foreg_cropd(self, image, label):
        def adjust(foreg_slice, patch_size, label, idx):
            diff = patch_size[idx - 1] - (foreg_slice[idx].stop - foreg_slice[idx].start)
            sign = -1 if diff < 0 else 1
            diff = abs(diff)
            ladj = self.randrange(diff)
            hadj = diff - ladj
            low = max(0, foreg_slice[idx].start - sign * ladj)
            high = min(label.shape[idx], foreg_slice[idx].stop + sign * hadj)
            diff = patch_size[idx - 1] - (high - low)
            if diff > 0 and low == 0:
                high += diff
            elif diff > 0:
                low -= diff
            return low, high

        cl = np.random.choice(np.unique(label[label > 0]))
        foreg_slices = scipy.ndimage.find_objects(_ndimage_label(label == cl)[0])
        foreg_slices = [x for x in foreg_slices if x is not None]
        slice_volumes = [np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices]
        slice_idx = np.argsort(slice_volumes)[-2:]
        foreg_slices = [foreg_slices[i] for i in slice_idx]
        if not foreg_slices:
            return self._rand_crop(image, label)
        foreg_slice = foreg_slices[random.randrange(len(foreg_slices))]
        low_x, high_x = adjust(foreg_slice, self.patch_size, label, 1)
        low_y, high_y = adjust(foreg_slice, self.patch_size, label, 2)
        low_z, high_z = adjust(foreg_slice, self.patch_size, label, 3)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]


class RandFlip:
    def __init__(self):
        self.axis = [1, 2, 3]
        self.prob = 1 / len(self.axis)

    def flip(self, data, axis):
        data["image"] = np.flip(data["image"], axis=axis).copy()
        data["label"] = np.flip(data["label"], axis=axis).copy()
        return data

    def __call__(self, data):
        for axis in self.axis:
            # if random.random() < self.prob:
            data = self.flip(data, axis)
        return data


class Cast:
    def __init__(self, types):
        self.types = types

    def __call__(self, data):
        data["image"] = data["image"].astype(self.types[0])
        data["label"] = data["label"].astype(self.types[1])
        return data


class RandomBrightnessAugmentation:
    def __init__(self, factor, prob):
        self.prob = prob
        self.factor = factor

    def __call__(self, data):
        image = data["image"]
        factor = np.random.uniform(low=1.0-self.factor, high=1.0+self.factor, size=1)
        image = (image * (1 + factor)).astype(image.dtype)
        data.update({"image": image})
        return data


class GaussianNoise:
    def __init__(self, mean, std, prob):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, data):
        image = data["image"]
        # for i in range(10):
        scale = np.random.uniform(low=0.0, high=self.std)
        noise = np.random.normal(loc=self.mean, scale=scale, size=image.shape).astype(image.dtype)
        data.update({"image": image + noise})
        return data


import time
import numpy as np
from torch.utils.data import Dataset
def _decode_migration(item):
    """Normalise whatever the front worker put on the slow queue.

    New protocol : (idx, resume_stage, image, label)
    Legacy        : idx                       -> restart from stage 0
    """
    if isinstance(item, (int, np.integer)):
        return int(item), 0, None, None
    idx, resume_stage, image, label = item
    return int(idx), int(resume_stage), image, label


def slow_worker(images, labels, patch_size, oversampling, transforms,
                slow_queue, slow_processed_queue, cfg=None, worker_id=0,
                stats_queue=None):
    """Background worker: resume a migrated sample at its transformation index.

    The sample arrives partially processed, together with the index of the
    stage that was in flight when the front worker gave up.  We re-execute
    only that stage and everything after it -- not the whole pipeline.
    """
    cfg = cfg or MinatoConfig()
    rand_crop = RandBalancedCrop(patch_size=patch_size, oversampling=oversampling)
    stages = build_stages(rand_crop, transforms)
    n_stages = len(stages)
    recorder = MigrationRecorder(cfg.profile_dir, "bg")

    while True:
        item = slow_queue.get()
        if item is None:
            break  # Signal to stop
        try:
            idx, resume_stage, image, label = _decode_migration(item)
            resume_stage = max(0, min(resume_stage, n_stages))
            if image is None:
                # Nothing had completed yet in the front worker (or a legacy
                # index-only message): fall back to the full pipeline.
                data = {"image": images[idx], "label": labels[idx]}
                resume_stage = 0
            else:
                data = {"image": image, "label": label}

            timings = []
            t0 = time.perf_counter()
            data = run_stages(stages, data, start_index=resume_stage,
                              checkpoint=None, abort_event=None, timings=timings)
            elapsed = time.perf_counter() - t0

            slow_processed_queue.put((data["image"], data["label"]))
            recorder.write("resume", idx, stage=resume_stage, n_stages=n_stages,
                           elapsed_s=elapsed, stage_times=timings)
            if stats_queue is not None:
                try:
                    stats_queue.put(("resume", idx, elapsed))
                except Exception:
                    pass
            if cfg.verbose:
                print("[minato/bg %d] idx=%d resumed at stage %d/%d (%s) in %.4fs"
                      % (worker_id, idx, resume_stage, n_stages,
                         stages[resume_stage][0] if resume_stage < n_stages else "-",
                         elapsed))
        except Exception as e:
            print(f"[Slow Worker] Failed processing item {item!r}: {e}")
            recorder.write("resume_fail", -1)

import time
import numpy as np
from torch.utils.data import Dataset
import concurrent.futures
import time

import os
import time
import queue
import threading
import multiprocessing as mp
import numpy as np
from torch.utils.data import Dataset
import concurrent.futures

class PytTrain(Dataset):
    """Front-end dataset.

    Two changes relative to the original artifact:

    * ``t_out`` is no longer the constant 2.0 s.  A warm-up profiling pass
      measures the end-to-end latency distribution of this workload and sets
      ``t_out`` to ``--calib_percentile`` (P75) of it.
    * A sample that exceeds ``t_out`` is not thrown away.  The pipeline runs
      as an indexed list of stages that checkpoints after each stage, so the
      migration payload is ``(idx, resume_stage, image, label)`` and the
      background worker resumes at ``resume_stage``.
    """

    def __init__(self, images, labels, slow_queue, slow_processed_queue,
                 train_transforms=None, **kwargs):
        self.images = images
        self.labels = labels
        self.train_transforms = train_transforms if train_transforms is not None \
            else get_train_transforms()
        self.patch_size = kwargs["patch_size"]
        self.oversampling = kwargs["oversampling"]
        self.rand_crop = RandBalancedCrop(self.patch_size, self.oversampling)

        self.slow_queue = slow_queue
        self.slow_processed_queue = slow_processed_queue

        self.cfg = kwargs.get("minato_cfg") or MinatoConfig(
            **{k: v for k, v in kwargs.items() if k in MinatoConfig.FIELDS})

        # Preload into RAM (or use mmap_mode="r" if too large)
        self.images_data = [np.load(f, mmap_mode="r") for f in self.images]
        self.labels_data = [np.load(f, mmap_mode="r") for f in self.labels]

        # The pipeline as an indexed list; position == resume index.
        self.stages = build_stages(self.rand_crop, self.train_transforms)
        self.n_stages = len(self.stages)

        self.recorder = MigrationRecorder(self.cfg.profile_dir, "front")

        # Thread pool is created lazily *per process*: DataProducer forks this
        # object, and an inherited pool with live threads deadlocks the child.
        self._executor = None
        self._executor_pid = None

        # ---- warm-up profiling pass -> workload-relative t_out -----------
        self.calibration = self._calibrate()
        self.sample_timeout = float(self.calibration["t_out"])

        # t_out lives in shared memory whether or not adaptation is on, so the
        # fixed and adaptive paths run identical per-sample code.
        self.threshold = SharedThreshold(self.sample_timeout,
                                         self.cfg.calib_percentile)
        self.stats_queue = mp.Queue() if self.cfg.adapt else None
        self.controller = None
        if self.cfg.adapt:
            self.controller = ThresholdController(
                self.threshold, self.stats_queue, self.cfg,
                profile_dir=self.cfg.profile_dir)
            self.controller.start()
            print("[minato/adapt] background profiler running: P%.0f, fallback "
                  "P%.0f at migration rate > %.2f, censoring=%s"
                  % (self.cfg.calib_percentile,
                     self.cfg.adapt_fallback_percentile,
                     self.cfg.adapt_escalate_rate, self.cfg.adapt_censoring))
        self.recorder.dump_calibration(self.calibration, "train")
        mllog_event(key="minato_t_out", value=self.sample_timeout, sync=False)
        mllog_event(key="minato_t_out_percentile",
                    value=self.cfg.calib_percentile, sync=False)

        # Start slow workers
        self.slow_workers = [
            mp.Process(
                target=slow_worker,
                args=(self.images_data, self.labels_data,
                      self.patch_size, self.oversampling,
                      self.train_transforms,
                      self.slow_queue, self.slow_processed_queue,
                      self.cfg, i, self.stats_queue)
            ) for i in range(max(0, int(self.cfg.num_slow_workers)))
        ]
        for w in self.slow_workers:
            w.daemon = True
            w.start()

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.images)

    def _get_executor(self):
        pid = os.getpid()
        if self._executor is None or self._executor_pid != pid:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, int(self.cfg.front_threads)),
                thread_name_prefix="minato-front-%d" % pid)
            self._executor_pid = pid
        return self._executor

    # ---- calibration --------------------------------------------------
    def _measure_sample(self, executor, idx):
        """One profiled pass over the full pipeline.

        Deliberately runs through the *same* thread pool, checkpointing and
        clock as ``__getitem__``.  Measuring the pipeline synchronously here
        and then comparing against a threaded run at training time biases
        t_out low by the thread-handoff cost, which on a tight latency
        distribution is enough to migrate every single sample.
        """
        timings = []
        state = {}
        started = threading.Event()

        def checkpoint(stage, data):
            state["stage"] = stage
            state["data"] = data

        def work():
            started.set()
            data = {"image": self.images_data[idx], "label": self.labels_data[idx]}
            return run_stages(self.stages, data, start_index=0,
                              checkpoint=checkpoint, abort_event=None,
                              timings=timings,
                              copy_checkpoints=self.cfg.checkpoint_copy)

        future = executor.submit(work)
        started.wait(timeout=self.cfg.queue_wait_timeout)
        t0 = time.perf_counter()
        future.result()
        return time.perf_counter() - t0, timings

    def _calibrate(self):
        # A throwaway pool: its threads are joined before DataProducer forks
        # this dataset, so no child inherits a half-dead executor.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(self.cfg.front_threads)),
            thread_name_prefix="minato-calib")
        try:
            calibrator = TimeoutCalibrator(self.cfg, self.stages, tag="train")
            result = calibrator.calibrate(
                lambda idx: self._measure_sample(executor, idx),
                n_dataset=len(self.images),
                seed=1234,
                extra={"patch_size": list(self.patch_size),
                       "oversampling": float(self.oversampling)})
        finally:
            executor.shutdown(wait=True)
        return _broadcast_t_out(result)

    # ---- the front path ------------------------------------------------
    def __getitem__(self, idx):
        # Re-read per sample: a rewritten threshold takes effect on the next
        # sample rather than the next epoch. A stale read is harmless.
        sample_timeout = self.threshold.get()
        state = {"stage": 0, "data": None}
        lock = threading.Lock()
        abort = threading.Event()
        started = threading.Event()
        timings = []

        def checkpoint(stage, data):
            with lock:
                state["stage"] = stage
                state["data"] = data

        def work():
            started.set()
            data = {"image": self.images_data[idx], "label": self.labels_data[idx]}
            return run_stages(self.stages, data, start_index=0,
                              checkpoint=checkpoint, abort_event=abort,
                              timings=timings,
                              copy_checkpoints=self.cfg.checkpoint_copy)

        submitted = time.perf_counter()
        future = self._get_executor().submit(work)
        # Do not start the deadline until the pool actually picks the task up,
        # otherwise queueing behind another sample looks like a slow sample.
        started.wait(timeout=self.cfg.queue_wait_timeout)
        queue_wait = time.perf_counter() - submitted
        t0 = time.perf_counter()

        try:
            data = future.result(timeout=sample_timeout)
            elapsed = time.perf_counter() - t0
            if self.stats_queue is not None:
                try:
                    self.stats_queue.put(("fast", idx, elapsed))
                except Exception:
                    pass
            if self.cfg.profile_fast_samples:
                self.recorder.write("fast", idx, stage=self.n_stages,
                                    n_stages=self.n_stages,
                                    queue_wait_s=queue_wait, elapsed_s=elapsed,
                                    preserved_s=elapsed, stage_times=timings)
            else:
                self.recorder.write("fast", idx, elapsed_s=elapsed)
            return data["image"], data["label"]

        except concurrent.futures.TimeoutError:
            # Migrate. Tell the in-flight stage to stop at the next boundary
            # and ship the last checkpoint, not the raw index.
            abort.set()
            future.add_done_callback(lambda f: f.exception())  # swallow
            elapsed = time.perf_counter() - t0
            with lock:
                stage = state["stage"]
                partial = state["data"]
            preserved = sum(dt for _, dt in timings[:stage])
            wasted = max(0.0, elapsed - preserved)

            if partial is None:
                # Stage 0 itself blew the budget: nothing to resume from.
                self.slow_queue.put((idx, 0, None, None))
            else:
                self.slow_queue.put((idx, stage, partial["image"], partial["label"]))

            if self.stats_queue is not None:
                try:
                    self.stats_queue.put(("migrate", idx, elapsed))
                except Exception:
                    pass
            self.recorder.write("migrate", idx, stage=stage, n_stages=self.n_stages,
                                queue_wait_s=queue_wait, elapsed_s=elapsed,
                                preserved_s=preserved, wasted_s=wasted,
                                stage_times=timings)
            if self.cfg.verbose:
                print("[minato/front] idx=%d migrated after %.4fs (t_out=%.4fs) "
                      "at stage %d/%d; %.4fs preserved, %.4fs re-executed"
                      % (idx, elapsed, sample_timeout, stage,
                         self.n_stages, preserved, wasted))
            return None

    # ---- teardown -------------------------------------------------------
    def shutdown(self):
        if getattr(self, "controller", None) is not None:
            self.controller.stop()
            self.controller.join(timeout=15)
            print("[minato/adapt] %r" % (self.controller.summary(),))
        for _ in getattr(self, "slow_workers", []):
            try:
                self.slow_queue.put(None)
            except Exception:
                pass
        for w in getattr(self, "slow_workers", []):
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()
        if self._executor is not None and self._executor_pid == os.getpid():
            self._executor.shutdown(wait=False)
        self.recorder.close()


def _broadcast_t_out(result):
    """Rank 0 profiles, everyone else reuses its t_out.

    Keeps the threshold identical across data-parallel ranks and avoids paying
    the warm-up cost N times.  Falls back to the local value if anything about
    the process group is unusual.
    """
    try:
        import torch
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return result
        if dist.get_world_size() < 2:
            return result
        backend = dist.get_backend()
        device = "cpu"
        if backend == "nccl" and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        tensor = torch.tensor([float(result["t_out"])], dtype=torch.float64,
                              device=device)
        dist.broadcast(tensor, src=0)
        agreed = float(tensor.item())
        if agreed != result["t_out"]:
            result = dict(result)
            result["t_out_local"] = result["t_out"]
            result["t_out"] = agreed
            result["source"] = result.get("source", "") + "+broadcast"
    except Exception as exc:  # pragma: no cover
        print("[minato/calib] broadcast skipped (%r); using local t_out" % (exc,))
    return result

class PytVal(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return np.load(self.images[idx]), np.load(self.labels[idx])






