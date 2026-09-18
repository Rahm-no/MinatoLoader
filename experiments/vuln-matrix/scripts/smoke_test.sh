#!/usr/bin/env bash
# Smoke test: synthetic corpus -> profile -> extract -> two loaders, one epoch.
# A few minutes. Must pass before the full matrix is triggered.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

WORK="${SMOKE_WORK:-${TMPDIR:-/tmp}/hmp_smoke}"
N_CORPUS="${N_CORPUS:-600}"
N_DATASET="${N_DATASET:-300}"
SPLITS="${SPLITS:-7F5B,3F9B}"
CONSUMER="${CONSUMER:-sleep:0.005}"

echo "== 0/5 torch-free selftest =="
python3 scripts/selftest.py

echo
echo "== 1/5 synthetic corpus ($N_CORPUS jpegs) -> $WORK/corpus =="
rm -rf "$WORK"; mkdir -p "$WORK"
python3 scripts/00_make_corpus.py --out "$WORK/corpus" -n "$N_CORPUS" --seed 0

echo
echo "== 2/5 profiling manifest =="
python3 scripts/01_profile_manifest.py \
    --root "$WORK/corpus" --pipeline synthetic_image \
    --out "$WORK/profile.csv" --seed 1234 --progress-every 200

echo
echo "== 3/5 extract mixed dataset (moderate, slow_fraction 0.25) =="
python3 scripts/02_extract_dataset.py \
    --profile "$WORK/profile.csv" --regime moderate \
    --slow-fraction 0.25 -n "$N_DATASET" \
    --dataset smoke --out "$WORK/mix.csv" --seed 1234

echo
echo "== 4/5 run pytorch + minato =="
python3 scripts/03_run_matrix.py \
    --manifest "$WORK/mix.csv" --pipeline synthetic_image \
    --splits "$SPLITS" --out-root "$WORK/results" \
    --batch-size 16 --baseline-workers 4 --warmup-samples 40 \
    --warmup-batches 2 --consumer "$CONSUMER" --tag smoke

echo
echo "== 5/5 validate the summary =="
python3 scripts/check_summary.py "$WORK/results/summary.csv"
echo
echo "smoke test complete. results in $WORK/results"
