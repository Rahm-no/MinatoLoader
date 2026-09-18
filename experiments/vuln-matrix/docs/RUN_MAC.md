# Mac: correctness validation only

This environment exists to prove the bundle runs end to end before you spend a
full matrix on the server. **Do not put Mac numbers in the paper.** No CUDA, no
`/dev/shm`, no CPU affinity — three of the methodology controls are simply
absent here.

## 1. Environment

```bash
cd experiments/vuln-matrix
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio        # CPU / MPS wheels
pip install pillow numpy psutil
```

`nvidia-ml-py` is not installed and not needed: `SystemMonitor` detects the
missing NVML and records CPU and memory only, leaving `gpu_util` blank. That is
a no-op, not a failure.

## 2. Selftest (seconds, no torch required)

```bash
python3 scripts/selftest.py
```

Must print `All checks passed.` This covers the pool/mix/resume/calibration/
metric logic. If it fails, stop — nothing downstream will be meaningful.

## 3. Smoke test (a few minutes)

```bash
bash scripts/smoke_test.sh
```

It builds a 600-image synthetic JPEG corpus, profiles it, extracts a 300-sample
mix at `slow_fraction 0.25` moderate, and runs PyTorch plus two Minato splits
for one epoch each. It finishes with `check_summary.py`, which fails loudly if
any required metric column is empty — that is the actual point of the exercise.

Knobs:

```bash
N_CORPUS=1200 N_DATASET=600 SPLITS=9F3B,5F7B CONSUMER=sleep:0.01 \
  bash scripts/smoke_test.sh
```

Expected output ends with `OK: every required field is populated.` plus one
line per run.

## 4. Optional: a RAM disk

macOS has no `/dev/shm`. The closest equivalent:

```bash
# 4 GB = 4*1024*2048 512-byte blocks
DEV=$(hdiutil attach -nomount ram://8388608)
diskutil erasevolume HFS+ RAMDisk "$DEV"
SMOKE_WORK=/Volumes/RAMDisk/hmp_smoke bash scripts/smoke_test.sh
# when finished
hdiutil detach "$DEV"
```

Worth doing only if you want the smoke test to be a rough rehearsal of the
staging step. It changes nothing about correctness.

## 5. Known Mac-specific behaviour

- **`--pin` is accepted and ignored.** `os.sched_setaffinity` does not exist on
  Darwin; `_pin()` returns immediately. Worker counts are the only concurrency
  control here.
- **`spawn` is the default start method**, so every worker re-imports the
  module tree. Startup is slower than on Linux and the stage objects are
  rebuilt per worker — intended, and the same code path the server uses for the
  DataLoader baseline.
- **`Queue.qsize()` raises `NotImplementedError` on macOS.** The queue tracer
  records `-1` rather than crashing, so `queue_trace.json` from a Mac run is
  not usable for vulnerability 3. Collect that on the server.
- If Activity Monitor shows the machine saturated, the migration rate will be
  inflated by CPU contention exactly as it would be on any small box. Ignore
  the rate here; you are checking that fields are populated, not what they say.
