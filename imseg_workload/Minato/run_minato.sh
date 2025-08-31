#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=$1   # first argument = number of GPUs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Root dir: $ROOT_DIR"

# Log file for GPU/CPU usage
USAGE_LOG="${SCRIPT_DIR}/cpu_gpu_usage_minato.csv"

# Start cpu_gpu_usage in background
"${ROOT_DIR}/cpu_gpu_usage.sh" "$NUM_GPUS" "$USAGE_LOG" &
USAGE_PID=$!

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    # Kill monitor
    kill $USAGE_PID 2>/dev/null || true

    # Kill anything this script launched (python, torchrun, sh)
    pkill -9 -f run_and_time.sh 2>/dev/null || true
    pkill -9 -f torchrun 2>/dev/null || true
    pkill -9 -f python 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_time=$(date +%s)

# Run training
"${SCRIPT_DIR}/run_and_time.sh" 1 "$NUM_GPUS"

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "GPU/CPU usage log: $USAGE_LOG"