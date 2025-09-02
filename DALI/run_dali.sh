#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <NUM_GPUS>"
    exit 1
fi

NUM_GPUS=$1     # first argument = number of GPUs
SYSTEM_NAME=dali  # second argument = system name (pytorch/dali/minato)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Root dir: $ROOT_DIR"
echo "System: $SYSTEM_NAME"
echo "GPUs: $NUM_GPUS"

# Create results directory for this system
RESULTS_DIR="${ROOT_DIR}/results/${SYSTEM_NAME}"
mkdir -p "$RESULTS_DIR"

# Log file for GPU/CPU usage (per system)
USAGE_LOG="${RESULTS_DIR}/gpu_cpu_usage.csv"

# Start cpu_gpu_usage in background
"${ROOT_DIR}/scripts/cpu_gpu_usage.sh" "$NUM_GPUS" "$USAGE_LOG" &
USAGE_PID=$!

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    kill $USAGE_PID 2>/dev/null || true
    pkill -9 -f run_and_time.sh 2>/dev/null || true
    pkill -9 -f torchrun 2>/dev/null || true
    pkill -9 -f python 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_time=$(date +%s)

# Run training (system-specific run_and_time.sh in the same folder)
"${SCRIPT_DIR}/run_and_time.sh" 1 "$NUM_GPUS"

end_time=$(date +%s)
duration=$((end_time - start_time))

