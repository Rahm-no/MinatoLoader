#!/bin/bash

# This script monitors CPU, memory, and GPU usage in real-time while excluding the lowest GPU from averages
echo -e "Time(s)\tCPU(%) GPU(%) LGPU(%)"

start_time=$(date +%s)

while true; do
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    
    # Get CPU usage (user CPU)
    cpuUsage=$(top -bn1 | awk '/Cpu/ {print $2}')
    
    # Get memory usage (in MB)
    memUsage=$(free -m | awk '/Mem/{print $3}')
    
    # Get GPU utilizations using nvidia-smi pmon. If a process's "mem" is 0, its SM% is treated as 0.
    gpuUsages=($(nvidia-smi pmon -s u -c 1 | awk '
    BEGIN { max_gpu = -1 }
    NR <= 2 { next }  # Skip headers
    {
        gpu = $1 + 0
        mem = $5 + 0  # Extract memory usage of the process
        sm = (mem == 0) ? 0 : $4 + 0  # Set SM% to 0 if mem is 0
        sum[gpu] += sm
        if (gpu > max_gpu) max_gpu = gpu
    }
    END {
        for (i = 0; i <= max_gpu; i++) {
            if (i in sum) {
                print sum[i]
            } else {
                print 0
            }
        }
    }'))
    
    # GPU Processing Logic
    if [ ${#gpuUsages[@]} -gt 0 ]; then
        # Only exclude lowest if we have multiple GPUs
        if [ ${#gpuUsages[@]} -gt 1 ]; then
            # Find lowest GPU
            lowest_usage=${gpuUsages[0]}
            lowest_index=0
            for i in "${!gpuUsages[@]}"; do
                if [ "${gpuUsages[i]}" -lt "$lowest_usage" ]; then
                    lowest_usage="${gpuUsages[i]}"
                    lowest_index=$i
                fi
            done
            
            # Create new array without lowest GPU
            filtered_gpus=()
            for i in "${!gpuUsages[@]}"; do
                [ $i -ne $lowest_index ] && filtered_gpus+=("${gpuUsages[i]}")
            done
            
            # Calculate average
            total=0
            for usage in "${filtered_gpus[@]}"; do
                total=$((total + usage))
            done
            gpuAverage=$((total / ${#filtered_gpus[@]}))
            lowestGpu=$lowest_usage
        else
            # Single GPU - use as-is
            gpuAverage=${gpuUsages[0]}
            lowestGpu=${gpuUsages[0]}
        fi
    else
        gpuAverage=0
        lowestGpu=0
    fi
    
    # Print output
    echo -e "${elapsed_time}\t${cpuUsage}\t${gpuAverage}\t${lowestGpu}"
    
    sleep 1
done