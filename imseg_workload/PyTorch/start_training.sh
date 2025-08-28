#!/bin/bash

SCRIPT_DIR=$(dirname -- "$(readlink -f -- "$0")")
echo $SCRIPT_DIR

CURRENT_USER=$(whoami)
echo $CURRENT_USER
echo $SCRIPT_DIR
ls -ld "$SCRIPT_DIR"
mkdir -p "${SCRIPT_DIR}/output"
mkdir -p "${SCRIPT_DIR}/ckpts"

NUM_GPUS=${1:-8}
CONTAINER_NAME=${2:-train_imseg}
BATCH_SIZE=${3:-2}
DOCKER_MEMORY=${4:-""}

DOCKER_MEMORY_PARAM=

if [ ! -z "$DOCKER_MEMORY" ]; then
    DOCKER_MEMORY_PARAM="-m $((DOCKER_MEMORY * 1024 * 1024 * 1024))"
fi



#!/bin/bash

CONTAINER_NAME=my_container
SCRIPT_DIR=$(pwd)  # or set manually if needed

sudo docker run -it --ipc=host --rm --runtime=nvidia \
  --name="$CONTAINER_NAME" \
  -v /raid/data/imseg/raw-data/kits19/preproc-data/:/data \
  -v "$(pwd)":/workspace \
  -v "${SCRIPT_DIR}/output":/results \
  -v "${SCRIPT_DIR}/ckpts":/ckpts \
  unet3d:rahma bash
# sudo docker run -it --ipc=host --rm --runtime=nvidia \
#   --name="$CONTAINER_NAME" \
#   # --memory=80g \
#   # --memory-swap=80g \
#   -v /raid/data/imseg/raw-data/kits19/data/:/raw_data \
#   -v "$(pwd)":/workspace \
#   -v /raid/data/imseg/raw-data/kits19/200gb_data/:/data \
#   -v "${SCRIPT_DIR}/output":/results \
#   -v "${SCRIPT_DIR}/ckpts":/ckpts \
#   unet3d:rahma bash

#-v /raid/data/unet3d/29gb-npy-prepp/:/data \
  # --memory=80g \
  # --memory-swap=150g \