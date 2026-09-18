#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5> <num_gpus>

SEED=${1:--1} 
NUM_GPUS=${2:-8}
BATCH_SIZE=${3:-2}
NUM_WORKERS=${4:-2}
NUM_EPOCHS=${5:-10}


QUALITY_THRESHOLD="0.908"
START_EVAL_AT=500
EVALUATE_EVERY=500
LEARNING_RATE="0.8"
LR_WARMUP_EPOCHS=10
DATASET_DIR="/data"
RAW_DATASET_DIR="/raw_data"
GRADIENT_ACCUMULATION_STEPS=1

# --- Minato: warm-up calibration of t_out -----------------------------------
# CALIB_SECONDS=0 disables profiling and falls back to SAMPLE_TIMEOUT.
CALIB_SECONDS=${CALIB_SECONDS:-600}
CALIB_PERCENTILE=${CALIB_PERCENTILE:-75}
CALIB_MARGIN=${CALIB_MARGIN:-1.0}
CALIB_CACHE=${CALIB_CACHE:-/results/minato_calibration.json}
PROFILE_DIR=${PROFILE_DIR:-/results/minato_profile}
SAMPLE_TIMEOUT=${SAMPLE_TIMEOUT:-2.0}
NUM_SLOW_WORKERS=${NUM_SLOW_WORKERS:-3}
if [ "${CALIB_SECONDS}" = "0" ]; then CALIBRATE=false; else CALIBRATE=true; fi
SAVE_CKPT_PATH="/ckpts"

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

python3 -c "import torch; print(torch.__version__)"




  torchrun  --nproc_per_node=${NUM_GPUS} main.py \
    --data_dir ${DATASET_DIR} \
    --raw_dir  ${RAW_DATASET_DIR} \
    --epochs  ${NUM_EPOCHS} \
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
    --num_workers ${NUM_WORKERS} \
    --calibrate ${CALIBRATE} \
    --calib_seconds ${CALIB_SECONDS} \
    --calib_percentile ${CALIB_PERCENTILE} \
    --calib_margin ${CALIB_MARGIN} \
    --calib_cache ${CALIB_CACHE} \
    --sample_timeout ${SAMPLE_TIMEOUT} \
    --num_slow_workers ${NUM_SLOW_WORKERS} \
    --profile_dir ${PROFILE_DIR}

    # end timing
    end=$(date +%s)


	# report result
	result=$(( $end - $start ))
	result_name="image_segmentation"
  


else
	echo "Directory ${DATASET_DIR} does not exist"
fi