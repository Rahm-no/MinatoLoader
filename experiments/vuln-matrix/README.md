# H-AIP vulnerability matrix: MinatoLoader vs PyTorch DataLoader

Sweeps `(slow_fraction × cost_ratio × F/B split)` for both loaders and records
the metrics needed to test three claimed architectural vulnerabilities of the
reactive fast/slow classification design.

Run against the **corrected** loader — calibrated `t_out` and resume-from-index,
both merged in the previous commit. Adaptive threshold adjustment and adaptive
worker scaling are deliberately **not** implemented; `t_out` is computed once
from warm-up and the F/B split is fixed per run, as agreed.

---

## Which environment runs what

| | Mac | Linux server (16 cores, Quadro) |
|---|---|---|
| `scripts/selftest.py` | yes | yes |
| `scripts/smoke_test.sh` | **yes — this is the Mac's job** | yes, before the matrix |
| `scripts/01_profile_manifest.py` | only on the synthetic corpus | yes, on the real datasets |
| `scripts/run_full_matrix.sh` | **no** | yes |
| GPU monitoring | no-op | NVML |
| CPU affinity | unavailable, `--pin` is a no-op | `--pin` |
| Staging | see `docs/RUN_MAC.md` | `/dev/shm` |

**No number from the Mac goes in the paper.** The Mac path exists to prove the
manifests load, both loaders run to completion, and every metric field is
populated before you spend a full matrix on the server.

---

## Files

```
hmp/manifest.py     percentile pools, deterministic mixing, sentinels   (torch-free)
hmp/minato.py       stage runner, warm-up calibration, F/B workers      (torch-free core)
hmp/pipelines.py    OpenImages / LibriSpeech / synthetic stage lists    (torch)
hmp/runtime.py      baseline loader, consumer, monitor, aggregation
scripts/00_make_corpus.py       synthetic JPEG corpus (smoke test only)
scripts/01_profile_manifest.py  script 1 — profiling manifest
scripts/02_extract_dataset.py   script 2 — extractor / mixer
scripts/03_run_matrix.py        script 3 — matrix runner
scripts/selftest.py             torch-free logic checks
scripts/smoke_test.sh           end-to-end smoke test
scripts/check_summary.py        fails if any metric field is empty
scripts/run_full_matrix.sh      the full sweep (Linux)
```

---

## Vulnerability → metric

| Claim | Columns in `summary.csv` |
|---|---|
| 1. Reactive detection tax scales with `slow_fraction`, independent of split | `detect_tax_total_s`, `detect_tax_per_sample_s`, `migrations`, `t_out_s`. Plot `detect_tax_per_sample_s` against `slow_fraction`, one line per split — the claim predicts the lines overlap. |
| 2. P75 is a minority-tail assumption | `slow_recall`, `fn_slow_kept_fast`, `misclassification_rate`, `warmup_slow_below_t_out_frac`. The claim predicts recall collapses once `slow_fraction > 0.25`. |
| 3. Fixed F/B split can't track a shifting requirement | `throughput_samples_s` and `batch_p95_s` across splits at fixed `slow_fraction`; queue occupancy in `queue_trace.json`. The claim predicts no split dominates. |

Two metrics the original paper does not report and which this harness adds:

- **Ground-truth misclassification.** Every sample carries the pool it was
  drawn from, so the runtime classification is scored against a known label
  rather than inferred.
- **Batch composition skew** (`early_batch_slow_frac`, `late_batch_slow_frac`,
  `batch_composition_skew`). Preferring the fast queue when building batches
  means early batches are systematically depleted of slow-pool samples. On
  OpenImages "slow" correlates with image resolution, so this is a shift in the
  *data distribution* across the epoch, not only a throughput effect. Worth a
  look even where throughput is a wash.

---

## Quick start

```bash
pip install -r requirements.txt
python3 scripts/selftest.py          # seconds, no torch needed
bash scripts/smoke_test.sh           # a few minutes
```

Then `docs/RUN_MAC.md` or `docs/RUN_LINUX.md`.

---

## Things you should know before reading any output

**The hypothesis is falsifiable and the harness lets it fail.** `summary.csv`
carries signed throughput for every `(config, loader)` pair, so a region where
Minato wins shows up as a win. Design the analysis to report the whole surface,
not the losing corner of it.

**Decode is stage 0 in both pipelines, and that blunts resume-from-index.**
OpenImages cost heterogeneity is dominated by JPEG decode; LibriSpeech's by
audio load and utterance length. Both are at the very front of the pipeline. If
`t_out` expires before stage 0 completes there is no checkpoint, the payload
carries `None`, and the background worker restarts from scratch. Watch
`migrations_at_stage_0`: if it is most of `migrations`, resume-from-index is
barely engaging on this workload and any conclusion about it is a conclusion
about stage granularity rather than about the mechanism. The OpenImages pipeline
already splits file-read from decode for exactly this reason; splitting decode
itself would need a progressive JPEG decoder.

**Sentinels cannot be compared across ratio regimes.** The fast pool is P0–P50
in both regimes so the fast sentinel is shared, but the slow pools (P75–P90 and
P97+) are disjoint by construction, so the slow sentinel is necessarily
per-regime. Compare slow sentinels within a regime only.

**The consumer matters.** Minato's claim is about overlapping slow-sample work
with GPU compute. Measured with no downstream consumer, there is nothing to
overlap with and the comparison is meaningless. Default is
`--consumer sleep:0.03`; use `--consumer model:resnet18` on the Quadro for the
paper numbers, and record which one produced each figure — it is in
`run_meta.json` and in the `consumer` column.

**LibriSpeech stage order was corrected.** The brief lists
`Pad → SpecAugment → FilterBank → …`. SpecAugment masks a mel spectrogram, so it
cannot precede the filterbank. Implemented as
`Pad → FilterBank → SpecAugment → FrameSplicing → Normalize` (NeMo/Jasper order).

**Sampling with replacement.** If you ask for more samples than a pool holds,
the mixer samples with replacement and records a warning in the manifest's
`.meta.json`. At `slow_fraction 0.60` with the extreme regime the slow pool is
only ~3% of candidates, so this will fire unless the profiling pool is large.
Profile at least `n_samples × slow_fraction / 0.03` candidates for that corner,
or accept repeats and say so.

---

## What has and has not been verified

Verified by execution (`scripts/selftest.py`, all passing):

- percentile pools; fast pool identical across regimes; slow pools disjoint;
  extreme ratio > moderate ratio
- exact `slow_fraction` proportions at 0.10 / 0.25 / 0.40 / 0.60; seed
  reproducibility; sentinel injection at every fraction
- stage-resume equivalence at every index; checkpoint isolation from in-place
  stages; abort loses exactly one stage
- `t_out` equals the measured P75; margin arithmetic; per-label warm-up
  breakdown
- **the vulnerability-2 mechanism in miniature**: at 70% slow, 64% of slow
  samples fall below the P75 cut
- the confusion matrix, detection-tax, and wasted-CPU arithmetic in the
  summariser, against hand-computed values

Not verified by execution — the torch paths. I could not install torch in the
environment I built this in (the CUDA dependency chain exhausted the disk), so
`hmp/pipelines.py`, the baseline DataLoader, the consumer and the NVML monitor
are compile-checked but never run. `scripts/smoke_test.sh` is the first thing
that exercises them; run it on the Mac before you touch the server, and treat
its first failure as expected rather than alarming.
