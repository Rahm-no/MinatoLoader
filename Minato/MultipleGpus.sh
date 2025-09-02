#!/bin/bash

# Specify the GPUs you want to monitor
GPU_LIST=(2 4 6 8)

for GPU_NUMBER in "${GPU_LIST[@]}"; do
    OUTPUT_FILE="/raid/mlcomns_imseg_1/results_without_randomness/sync_dataloader/training_${GPU_NUMBER}.txt"
    GPU_LOG_FILE="/raid/mlcomns_imseg_1/results_without_randomness/sync_dataloader/gpu_${GPU_NUMBER}.txt"

    # Run start_training.sh with the current GPU_NUMBER
    ./start_training.sh $GPU_NUMBER train_imseg 4 "" > $OUTPUT_FILE 2>&1 &

    # Capture the process ID of the last background command (start_training.sh)
    TRAINING_PID=$!

    # Run the GPU monitoring script in the background
    ./gpu_script.sh > $GPU_LOG_FILE 2>&1 &

    # Wait for the training script to finish before moving on to the next GPU
    wait $TRAINING_PID

    # Capture the process ID of the GPU monitoring script
    GPU_MONITORING_PID=$(pgrep -f "gpu_script.sh")

    # Terminate the GPU monitoring script
    kill $GPU_MONITORING_PID

    echo "Training with GPU $GPU_NUMBER completed. Output saved to $OUTPUT_FILE"
done
