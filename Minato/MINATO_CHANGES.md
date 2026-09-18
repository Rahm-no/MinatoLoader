# Calibrated `t_out` and resume-from-index migration

Two behaviours described in the MinatoLoader paper were missing from the
artifact. This patch adds them.

## 1. `t_out` is calibrated from the workload

**Before.** `PytTrain.__init__` set `self.sample_timeout = 2`. Every sample
that took longer than two seconds on the front path was declared slow,
regardless of what the rest of the dataset looked like. On a workload whose
median sample takes 200 ms the threshold never fires; on one whose median takes
3 s everything migrates.

**After.** `PytTrain` runs a warm-up profiling pass before training starts. It
times a random subset of the real dataset through the real pipeline and sets

```
t_out = percentile(latencies, --calib_percentile) * --calib_margin
```

with `--calib_percentile 75` by default. The pass stops at whichever comes
first: `--calib_seconds` (default 600, i.e. ten minutes) or
`--calib_max_samples` (default 256). The first `--calib_warmup_samples`
timings are discarded so a cold page cache does not skew the distribution.

Results are written to `--profile_dir` as `calibration_train_<pid>.json` and
can be cached in `--calib_cache` so that repeated runs on the same machine and
dataset skip the ten-minute pass. `--recalibrate` forces a fresh pass.

Under `torchrun`, rank 0 profiles and broadcasts its `t_out`; the other ranks
reuse it rather than each paying the warm-up cost. If the process group is not
initialised the local value is used.

Setting `--calibrate false` restores a fixed threshold, taken from
`--sample_timeout` (default 2.0), which reproduces the original behaviour.

### Caveat worth knowing

P75 is a *definition* of slow, not a diagnosis: by construction roughly 25% of
samples will exceed it and migrate. If your intent is "catch the pathological
tail", P90/P95 or `--calib_margin 1.3` is the knob. This matters most when the
latency distribution is bimodal — a fast mode holding 80% of samples and a slow
mode holding 20% puts P75 right at the upper edge of the fast mode, where small
amounts of CPU contention flip large numbers of samples across the threshold.
The report prints the migration rate so you can see this happening.

## 2. Migration carries partial state

**Before.** The pipeline ran as an opaque closure submitted to a
`ThreadPoolExecutor`. On timeout the front worker put the bare *index* on
`slow_queue` and returned `None`; the background worker then re-ran
`RandBalancedCrop` and every transform from scratch. Two costs followed:
everything the front worker had already done was discarded, and the orphaned
future kept running to completion in the front worker's pool because nothing
told it to stop.

**After.** The pipeline is an indexed list of stages built by `build_stages`:
index 0 is `RandBalancedCrop`, indices 1..N-1 are the entries of the
`transforms.Compose`. `run_stages` executes them, publishing a private copy of
the sample after each one and checking an abort flag at every boundary.

On timeout the front worker sets the abort flag and puts
`(idx, resume_stage, image, label)` on `slow_queue`. The background worker
resumes at `resume_stage`, so it re-executes only the stage that was in flight
plus the stages after it. The abandoned stage finishes in the front worker's
thread — Python threads cannot be preempted — and that single stage is the
wasted CPU per migration.

Two edge cases:

* If the *first* stage blows the budget there is no checkpoint yet, so the
  payload carries `None` and the background worker starts from scratch. The
  report counts these separately (`migrations_at_stage_0`). If they dominate,
  your slowest stage is the first one and resume-from-index cannot help until
  that stage is split.
* Checkpoints are deep copies (`--checkpoint_copy`, on by default) because
  transforms such as `HeavyGaussianBlur` write into their input array. Without
  the copy a still-running stage would corrupt the state being migrated. The
  cost is one array copy per stage boundary on the front path only; set
  `--checkpoint_copy false` to measure it.

The payload is now an array rather than an integer, so migration costs a
pickle of the patch (a few MB) over the queue instead of a few bytes. That is
the trade for not recomputing the pipeline.

## Instrumentation

With `--profile_dir` set, each process appends to
`events_{front,bg}_<pid>.csv`: one row per fast sample, migration and resume,
with per-stage timings. `tools/minato_report.py <profile_dir>` aggregates them
and prints the calibrated threshold, the migration rate, where in the pipeline
migrations happen, and the wasted CPU under resume-from-index against the
counterfactual cost of the original full-restart scheme.

## New flags

| flag | default | meaning |
| --- | --- | --- |
| `--calibrate` | `true` | run the warm-up profiling pass |
| `--calib_seconds` | `600` | wall-clock budget for the pass |
| `--calib_max_samples` | `256` | sample budget for the pass |
| `--calib_min_samples` | `24` | do not stop on time before this many |
| `--calib_warmup_samples` | `3` | discarded leading timings |
| `--calib_percentile` | `75` | percentile used as `t_out` |
| `--calib_margin` | `1.0` | multiplier on the percentile |
| `--calib_cache` | `""` | JSON file to persist/reuse the result |
| `--recalibrate` | off | ignore the cache |
| `--sample_timeout` | `2.0` | fallback when calibration is off |
| `--t_out_min` / `--t_out_max` | `0.05` / `120` | clamp on the calibrated value |
| `--checkpoint_copy` | `true` | copy the sample at each stage boundary |
| `--front_threads` | `2` | threads in the front worker pool |
| `--num_slow_workers` | `3` | background resume processes (`0` = none) |
| `--profile_dir` | `""` | where to write calibration and event files |
| `--profile_fast_samples` | off | also record per-stage timings for fast samples |
| `--minato_verbose` | off | per-sample migration logging |

Every flag also reads an environment variable (`MINATO_CALIB_SECONDS`,
`MINATO_PROFILE_DIR`, ...) so the loader can be configured without touching
the launcher.

## Testing

`python3 tools/smoke_test.py` needs no GPU, no torchrun and no KiTS19 data.
Phases 1–3 are deterministic correctness checks; phase 4 runs a heavy-tailed
synthetic workload end to end and prints the report. On a machine with fewer
than four usable cores phase 4 is contention-bound and its rate-sensitive
checks are reported as warnings rather than failures.
