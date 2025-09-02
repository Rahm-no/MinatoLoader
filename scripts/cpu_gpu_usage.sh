#!/usr/bin/env bash
# gpu_cpu_logger.sh
# Usage: ./gpu_cpu_logger.sh NUM_GPUS
# Example: ./gpu_cpu_logger.sh 2   → logs average over GPU0 and GPU1
# Output file: gpu_cpu_log.csv

set -euo pipefail
LC_ALL=C

NUM_GPUS="${1:-8}"
USAGE_LOG="${2:-gpu_cpu_log_dali.csv}"
INTERVAL=1   # seconds

# Discover all GPU IDs
mapfile -t ALL_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)

# Take the first NUM_GPUS
GPU_IDS=("${ALL_GPUS[@]:0:$NUM_GPUS}")

# CSV header (only average, not per-GPU)
echo "elapsed_s,avg_cpu,gpu_avg" > "$USAGE_LOG"

start_epoch=$(date +%s)

# Seed CPU counters
read -r _ user nice system idle iowait irq softirq steal guest guest_nice < /proc/stat
prev_idle=$((idle + iowait))
prev_total=$((idle + iowait + user + nice + system + irq + softirq + steal))


while :; do
  sleep "$INTERVAL"

  # ----- CPU instantaneous percent -----
  read -r _ user nice system idle iowait irq softirq steal guest guest_nice < /proc/stat
  idle_all=$((idle + iowait))
  total=$((idle_all + user + nice + system + irq + softirq + steal))

  diff_total=$((total - prev_total))
  diff_idle=$((idle_all - prev_idle))
  cpu_pct="0.00"
  if (( diff_total > 0 )); then
    cpu_pct=$(awk -v dt="$diff_total" -v di="$diff_idle" 'BEGIN{printf "%.2f", (1 - di/dt)*100}')
  fi
  prev_total=$total; prev_idle=$idle_all

  # ----- GPU utilization (average across selected GPUs) -----
  sum=0
  for id in "${GPU_IDS[@]}"; do
    util=$(nvidia-smi -i "$id" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}')
    util=${util:-0}
    sum=$((sum + util))
  done

  avg=$(awk -v s="$sum" -v n="${#GPU_IDS[@]}" 'BEGIN{printf "%.2f", s/n}')

  # ----- Write CSV row -----
  elapsed=$(( $(date +%s) - start_epoch ))
  echo "$elapsed,$cpu_pct,$avg" >> "$USAGE_LOG"
done
