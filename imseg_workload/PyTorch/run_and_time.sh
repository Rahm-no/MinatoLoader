#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5> <num_gpus>

SEED=${1:--1}
ddplaunch=$(python -c "from os import path; import torch; print(path.join(path.dirname(torch.__file__), 'distributed', 'launch.py'))")
NUM_WORKERS=${3:-4}  # Default to 4 workers if not provided

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

if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

# CLEAR YOUR CACHE HERE
  python -c "
from mlperf_logging.mllog import constants
from runtime.logging import mllog_event
mllog_event(key=constants.CACHE_CLEAR, value=True)"

  python $ddplaunch --nnode=1 --node_rank=0 --nproc_per_node=${NUM_GPUS} main.py \
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
    --num_workers ${NUM_WORKERS}

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="image_segmentation"


	echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi