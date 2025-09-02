#!/usr/bin/env bash
# run_all.sh
# Usage: ./scripts/run_all.sh <NUM_GPUS>
# Example: ./scripts/run_all.sh 8

set -euo pipefail

NUM_GPUS=${1:-8}   # default = 8 if not provided
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"

# List of systems to run
SYSTEMS=("DALI" "PyTorch" "Minato")

for sys in "${SYSTEMS[@]}"; do
    echo "===== Running $sys with $NUM_GPUS GPUs ====="

    cd "${ROOT}/${sys}"

    # Each run_* script should exist in the system directory
    bash "run_${sys,,}.sh" "$NUM_GPUS"

    echo "===== Finished $sys ====="
    cd "$ROOT"
done

echo "✅ All systems finished. Combined results are in $ROOT/results_allsystems.csv"
