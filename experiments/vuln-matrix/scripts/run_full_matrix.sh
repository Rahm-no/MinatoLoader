#!/usr/bin/env bash
# Full matrix. Linux server only -- do not run this on the Mac.
#
#   OpenImages : 2 regimes x 4 slow_fractions x (5 Minato splits + 1 baseline) = 48 runs
#   LibriSpeech: 1 regime  x 4 slow_fractions x (3 Minato splits + 1 baseline) = 16 runs
#
# Expects the two profiling manifests to exist already (step 1 of RUN_LINUX.md).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

MANIFEST_DIR="${MANIFEST_DIR:-manifests}"
RESULT_DIR="${RESULT_DIR:-results}"
SEED="${SEED:-1234}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CONSUMER="${CONSUMER:-model:resnet18}"
N_IMAGE="${N_IMAGE:-20000}"
N_AUDIO="${N_AUDIO:-8000}"
FRACTIONS="${FRACTIONS:-0.10 0.25 0.40 0.60}"
IMAGE_SPLITS="${IMAGE_SPLITS:-11F1B,9F3B,7F5B,5F7B,3F9B}"
AUDIO_SPLITS="${AUDIO_SPLITS:-9F3B,7F5B,5F7B}"
PIN="${PIN:---pin}"

run_group () {
  local ds="$1" pipeline="$2" profile="$3" regime="$4" n="$5" splits="$6"
  for frac in $FRACTIONS; do
    local mix="${MANIFEST_DIR}/${ds}_${regime}_f${frac}_n${n}.csv"
    echo ""
    echo "############ ${ds} ${regime} slow_fraction=${frac} ############"
    python3 scripts/02_extract_dataset.py \
        --profile "$profile" --regime "$regime" --slow-fraction "$frac" \
        -n "$n" --dataset "$ds" --out "$mix" --seed "$SEED"
    python3 scripts/03_run_matrix.py \
        --manifest "$mix" --pipeline "$pipeline" \
        --loaders pytorch,minato --splits "$splits" \
        --out-root "${RESULT_DIR}/${ds}" \
        --batch-size "$BATCH_SIZE" --baseline-workers 12 \
        --consumer "$CONSUMER" --warmup-samples 300 --warmup-batches 5 \
        --seed "$SEED" $PIN --tag "${regime}_f${frac}"
  done
}

for regime in moderate extreme; do
  run_group openimages openimages \
            "${MANIFEST_DIR}/openimages_profile.csv" "$regime" "$N_IMAGE" "$IMAGE_SPLITS"
done

# Confirmatory subset only, per the agreed scoping: moderate regime, 3 splits.
run_group librispeech librispeech \
          "${MANIFEST_DIR}/librispeech_profile.csv" moderate "$N_AUDIO" "$AUDIO_SPLITS"

echo ""
echo "matrix complete."
echo "  ${RESULT_DIR}/openimages/summary.csv"
echo "  ${RESULT_DIR}/librispeech/summary.csv"
