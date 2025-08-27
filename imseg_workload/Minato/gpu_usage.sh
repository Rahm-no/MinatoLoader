#!/bin/bash

echo -e "Time(s)\tCPU Avg\tGPU Avg\tGPU Power Avg(W)"

start_time=$(date +%s)

total_cpu=0
cpu_count=0
total_gpu=0
gpu_count=0
total_power=0
power_count=0

while true; do
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))

    # Get CPU usage (User CPU %)
    cpuUsage=$(top -bn1 | awk '/Cpu/ {print $2}')
    total_cpu=$(echo "$total_cpu + $cpuUsage" | bc)
    cpu_count=$((cpu_count + 1))
    avg_cpu=$(echo "scale=2; $total_cpu / $cpu_count" | bc)

    # Get GPU utilization and power draw
    gpuUsages=($(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits))
    gpuPowers=($(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits))
    gpuMemories=($(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits))

    if [ ${#gpuUsages[@]} -gt 0 ]; then
        running_gpu_count=0
        total_gpu_usage=0
        total_gpu_power=0

        for i in "${!gpuUsages[@]}"; do
            # If GPU memory is 0 and utilization is 99% or 100%, set usage to 0
            if [ "${gpuMemories[i]}" -eq 0 ] && ([ "${gpuUsages[i]}" -eq 99 ] || [ "${gpuUsages[i]}" -eq 100 ]); then
                gpuUsages[i]=0
            fi
            total_gpu_usage=$(echo "$total_gpu_usage + ${gpuUsages[i]}" | bc)
            total_gpu_power=$(echo "$total_gpu_power + ${gpuPowers[i]}" | bc)
            running_gpu_count=$((running_gpu_count + 1))
        done

        if [ "$running_gpu_count" -gt 0 ]; then
            avg_gpu_usage=$(echo "scale=2; $total_gpu_usage / $running_gpu_count" | bc)
            avg_gpu_power=$(echo "scale=2; $total_gpu_power / $running_gpu_count" | bc)
        else
            avg_gpu_usage="0"
            avg_gpu_power="0"
        fi

        total_gpu=$(echo "$total_gpu + $avg_gpu_usage" | bc)
        gpu_count=$((gpu_count + 1))
        total_power=$(echo "$total_power + $avg_gpu_power" | bc)
        power_count=$((power_count + 1))
    else
        avg_gpu_usage="0"
        avg_gpu_power="0"
    fi

    echo -e "${elapsed_time}s\t${avg_cpu}\t${avg_gpu_usage}\t${avg_gpu_power}"
    sleep 1
done