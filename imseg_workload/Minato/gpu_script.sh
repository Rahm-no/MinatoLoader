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

    # Get GPU SM% and MEM% for each GPU using nvidia-smi pmon
    gpuSmMem=($(nvidia-smi pmon -s um -c 1 | awk '
    BEGIN { OFS=" "; max_gpu = -1 }
    NR <= 2 { next }
    {
        gpu = $1 + 0
        sm = $4 + 0
        mem = $5 + 0

        # If mem == 0 and sm == 99, treat sm as 0
        if (mem == 0 && sm == 99) sm = 0

        sum_sm[gpu] += sm
        if (gpu > max_gpu) max_gpu = gpu
    }
    END {
        for (i = 0; i <= max_gpu; i++) {
            if (i in sum_sm) {
                print sum_sm[i]
            } else {
                print 0
            }
        }
    }'))

    # GPU Processing Logic
    if [ ${#gpuSmMem[@]} -gt 0 ]; then
        if [ ${#gpuSmMem[@]} -gt 1 ]; then
            # Find lowest GPU
            lowest_usage=${gpuSmMem[0]}
            lowest_index=0
            for i in "${!gpuSmMem[@]}"; do
                if [ "${gpuSmMem[i]}" -lt "$lowest_usage" ]; then
                    lowest_usage="${gpuSmMem[i]}"
                    lowest_index=$i
                fi
            done

            # Create new array without lowest GPU
            filtered_gpus=()
            for i in "${!gpuSmMem[@]}"; do
                [ $i -ne $lowest_index ] && filtered_gpus+=("${gpuSmMem[i]}")
            done

            # Calculate average
            total=0
            for usage in "${filtered_gpus[@]}"; do
                total=$((total + usage))
            done
            gpuAverage=$((total / ${#filtered_gpus[@]}))
            lowestGpu=$lowest_usage
        else
            gpuAverage=${gpuSmMem[0]}
            lowestGpu=${gpuSmMem[0]}
        fi
    else
        gpuAverage=0
        lowestGpu=0
    fi

    # Print output
    echo -e "${elapsed_time}\t${cpuUsage}\t${gpuAverage}\t${lowestGpu}"

    sleep 1
done
