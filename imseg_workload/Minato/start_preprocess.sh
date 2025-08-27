#!/bin/bash

SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )
mkdir -p ${SCRIPT_DIR}/output
mkdir -p ${SCRIPT_DIR}/ckpts


CONTAINER_NAME=${2:train_imseg}

DOCKER_MEMORY=${4:-""}

DOCKER_MEMORY_PARAM=

if [ ! -z "$DOCKER_MEMORY" ]
then
	DOCKER_MEMORY_PARAM="-m ${DOCKER_MEMORY}g"
fi



sudo docker run -it --ipc=host --rm --runtime=nvidia \
  --name="$CONTAINER_NAME" \
  --memory=100g \
  --memory-swap=200g \
  -v /raid/data/imseg/raw-data/kits19/data/:/raw_data \
  -v "$(pwd)":/workspace \
  -v /raid/data/imseg/raw-data/kits19/200gb_data/:/data \
  -v "${SCRIPT_DIR}/output":/results \
  -v "${SCRIPT_DIR}/ckpts":/ckpts \
  unet3d:rahma bash
