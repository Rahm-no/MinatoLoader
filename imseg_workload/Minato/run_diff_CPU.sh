#!/bin/bash
set -euo pipefail

# Usage: ./run_diff_workers.sh <seed> <num_gpus> "<workers list e.g. 2 4 8 16>"
SEED=${1:-1}
NUM_GPUS=${2:-8}
WORKERS_LIST=${3:-"2 16 24 32 48 64 72 80 96"}
LOG_DIR="diff_CPUs"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONTAINER_NAME="imseg_run"

mkdir -p "$LOG_DIR"

for NUM_WORKERS in $WORKERS_LIST; do
  echo "======================================"
  echo " Running with num_workers=$NUM_WORKERS"
  echo "======================================"

  GPU_LOG_FILE="$LOG_DIR/${NUM_WORKERS}_workers.csv"

  # Start GPU logging in the background
  ./gpu_script.sh >"$GPU_LOG_FILE" 2>&1 &
  GPU_MONITOR_PID=$!

  # Start training container in the background
  docker run --rm --gpus all \
    --runtime=nvidia \
    --ipc=host \
    --pid=host \
    --name "$CONTAINER_NAME" \
    -v /raid/data/imseg/raw-data/kits19/data/:/raw_data \
    -v /raid/data/imseg/raw-data/kits19/preproc-data/:/data \
    -v "$SCRIPT_DIR":/workspace \
    -v "$SCRIPT_DIR/output":/results \
    -v "$SCRIPT_DIR/ckpts":/ckpts \
    -w /workspace \
    unet3d:speedy \
    bash -c "./run_and_time.sh $SEED $NUM_GPUS 2 $NUM_WORKERS" &
  DOCKER_PID=$!

  # Wait 100 seconds
  sleep 100

  echo " Stopping training and GPU logging after 100s..."

  # Kill both processes
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  kill "$DOCKER_PID" 2>/dev/null || true

  # Wait to ensure clean exit
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$DOCKER_PID" 2>/dev/null || true

  echo "→ Completed num_workers=$NUM_WORKERS; log saved to $GPU_LOG_FILE"
  echo
done

echo "All runs complete. GPU logs are in $LOG_DIR/"
