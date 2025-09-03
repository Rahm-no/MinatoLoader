#!/bin/bash
set -e

SEED=${1:--1} 
NUM_GPUS=${2:-8}
MAX_EPOCHS=10
QUALITY_THRESHOLD="0.908"
START_EVAL_AT=50
EVALUATE_EVERY=50
LEARNING_RATE="0.8"
LR_WARMUP_EPOCHS=10
DATASET_DIR="/data"
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
SAVE_CKPT_PATH="/ckpts"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Root dir: $ROOT_DIR"

result_file="${ROOT_DIR}/results/results_allsystems.csv"   # <-- FIXED


if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

    # CLEAR YOUR CACHE HERE
    python3 -c "
from mlperf_logging.mllog import constants
from runtime.logging import mllog_event
mllog_event(key=constants.CACHE_CLEAR, value=True)"

    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --data_dir ${DATASET_DIR} \
        --epochs ${MAX_EPOCHS} \
        --evaluate_every ${EVALUATE_EVERY} \
        --start_eval_at ${START_EVAL_AT} \
        --quality_threshold ${QUALITY_THRESHOLD} \
        --batch_size ${BATCH_SIZE} \
        --optimizer sgd \
        --ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate ${LEARNING_RATE} \
        --seed ${SEED} \
        --lr_warmup_epochs ${LR_WARMUP_EPOCHS} \
        --save_ckpt_path ${SAVE_CKPT_PATH} \
        --num_workers 2

    # end timing
    end=$(date +%s)
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo "ENDING TIMING RUN AT $end_fmt"

    # report result
    result=$(( end - start ))
    result_name="image_segmentation"
    line="$end_fmt,DALI,$result"

    # create results file with header if not exists
    if [ ! -f "$result_file" ]; then
        echo "timestamp,system,seconds" > "$result_file"
    fi

    # append row
    echo "$line" >> "$result_file"

else
    echo "Directory ${DATASET_DIR} does not exist"
fi
