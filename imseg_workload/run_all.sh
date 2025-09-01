#!/usr/bin/env bash
# run_all.sh
# Usage: ./run_all.sh NUM_GPUS

set -euo pipefail

NUM_GPUS=${1:-8}
USAGE_LOG="USAGE-log"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# List of systems to run
SYSTEMS=("DALI" "PyTorch" "Minato")

for sys in "${SYSTEMS[@]}"; do
    echo "===== Running $sys with $NUM_GPUS GPUs ====="
    cd "${ROOT}/${sys}"

    # Run the workload
    bash "run_${sys,,}.sh" "$NUM_GPUS"

    echo "===== Finished $sys ====="
    cd "$ROOT"
done
