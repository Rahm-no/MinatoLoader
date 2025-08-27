#!/bin/bash

SCRIPT_DIR=$(dirname -- "$(readlink -f -- "$0")")
echo $SCRIPT_DIR

CURRENT_USER=$(whoami)
echo $CURRENT_USER
echo $SCRIPT_DIR
ls -ld "$SCRIPT_DIR"
mkdir -p "${SCRIPT_DIR}/output"
mkdir -p "${SCRIPT_DIR}/ckpts"

NUM_GPUS=${1:-1}
CONTAINER_NAME=${2:-train_imseg}
BATCH_SIZE=${3:-2}



sudo docker run -it --ipc=host --rm --runtime=nvidia \
  --name="$CONTAINER_NAME" \
  -v /raid/data/imseg/raw-data/kits19/data/:/raw_data \
  -v "$(pwd)":/workspace \
  -v /raid/data/imseg/raw-data/kits19/preproc-data/:/data \
  -v "${SCRIPT_DIR}/output":/results \
  -v "${SCRIPT_DIR}/ckpts":/ckpts \
  unet3d:speedy bash




#-v /raid/data/unet3d/29gb-npy-prepp/:/data \
