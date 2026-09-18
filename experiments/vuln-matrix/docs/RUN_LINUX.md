# Linux server (16 cores, Quadro): the real run

Everything destined for the paper comes from here.

## 1. Environment

```bash
cd experiments/vuln-matrix
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# match the CUDA build to the driver -- see pytorch.org/get-started/locally
pip install torch torchvision torchaudio
pip install pillow numpy psutil nvidia-ml-py
nvidia-smi                     # confirm the Quadro is visible
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Stage the datasets in DRAM

```bash
df -h /dev/shm                 # needs to hold the candidate pools
mkdir -p /dev/shm/hmp/{openimages,librispeech}
rsync -a --info=progress2 /path/to/openimages/subset/  /dev/shm/hmp/openimages/
rsync -a --info=progress2 /path/to/LibriSpeech/train-clean-100/ /dev/shm/hmp/librispeech/
```

`/dev/shm` defaults to half of RAM; raise it if the pools do not fit:

```bash
sudo mount -o remount,size=64G /dev/shm
```

Sizing the candidate pool: the extreme regime's slow pool is only the top ~3%
of candidates, so at `slow_fraction 0.60` and `n=20000` you need roughly
`20000 * 0.6 / 0.03 ≈ 400k` candidates to draw without replacement. Below that
the mixer samples with replacement and says so in the manifest metadata — an
acceptable compromise if you record it, a silent confound if you do not.

## 3. Quiet the machine

```bash
sudo cpupower frequency-set -g performance      # stop governor drift mid-sweep
sudo sysctl -w vm.drop_caches=3                 # once, before profiling
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

`OMP_NUM_THREADS=1` matters: without it every worker spawns its own OpenMP
pool and the F/B split stops meaning what you think it means.

## 4. Profiling manifests (script 1) — once per dataset

```bash
mkdir -p manifests
python3 scripts/01_profile_manifest.py \
    --root /dev/shm/hmp/openimages --pipeline openimages \
    --out manifests/openimages_profile.csv --seed 1234

python3 scripts/01_profile_manifest.py \
    --root /dev/shm/hmp/librispeech --pipeline librispeech \
    --out manifests/librispeech_profile.csv --seed 1234
```

Single-threaded by design — this pass defines the latency distribution that the
pools are cut from, so it must not be polluted by contention. It prints the
percentiles and the observed cost ratio per regime. **Check those ratios against
the design targets (4–6× moderate, 15–20× extreme) before going further.** If
they are off, the percentile cuts in `hmp/manifest.py:RATIO_REGIMES` need
adjusting for your data, and that is a decision to make deliberately rather than
discover in the results.

Re-run with `--repeats 3` if the machine is noisy; the median of three is used.

## 5. Smoke test on the server too

```bash
bash scripts/smoke_test.sh
```

Same few minutes, and it catches CUDA/NVML problems the Mac cannot.

## 6. Full matrix

```bash
tmux new -s matrix
CONSUMER=model:resnet18 N_IMAGE=20000 N_AUDIO=8000 \
  bash scripts/run_full_matrix.sh 2>&1 | tee results/matrix.log
```

64 runs: OpenImages 2 regimes × 4 fractions × (5 splits + baseline) = 48;
LibriSpeech moderate only × 4 fractions × (3 splits + baseline) = 16.

Per run you get `results/<dataset>/<run_id>/` containing `events/` (per-process
per-sample CSVs), `batches.csv`, `batches_warmup.csv`, `queue_trace.json`,
`system.csv`, `calibration.json`, `run_meta.json` — plus one appended row in
`results/<dataset>/summary.csv`.

## 7. Core budget

With `--pin`, `plan_cores` reserves 2 cores for the consumer and monitor and
deals the rest to the workers. At 16 cores that leaves 14 for a 12-worker split,
so `11F1B` and `9F3B` fit comfortably and nothing is oversubscribed. The PyTorch
baseline uses 12 pooled workers and is not pinned — `DataLoader` manages its own
workers — which is a real asymmetry between the two loaders. Note it in the
methodology; do not try to paper over it.

## 8. Analysis

```bash
python3 scripts/check_summary.py results/openimages/summary.csv
```

Then, per vulnerability:

1. `detect_tax_per_sample_s` vs `slow_fraction`, one line per split. Overlapping
   lines support the claim; separated lines refute it.
2. `slow_recall` and `fn_slow_kept_fast` vs `slow_fraction`. The predicted knee
   is at 0.25.
3. `throughput_samples_s` and `batch_p95_s` across splits at fixed
   `slow_fraction`, against the PyTorch baseline row for the same config. Report
   the whole surface including regions where Minato wins.

Also plot `batch_composition_skew` vs `slow_fraction` — it is the metric the
paper does not report, and it is a distributional claim rather than a
performance one.

## 9. Repeats

One epoch per config is one sample. Before anything goes in the paper, re-run
the matrix with `SEED=2234` and `SEED=3234` and report the spread. Your prior
9F+3B vs 7F+5B result was a 0.2–0.4% throughput difference; that is well inside
run-to-run noise for a dataloader benchmark, and a single-seed version of this
sweep will not be able to tell a 0.3% effect from nothing.
