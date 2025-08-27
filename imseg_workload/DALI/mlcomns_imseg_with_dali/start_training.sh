#!/bin/bash
echo $(whoami) 
SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )
echo $SCRIPT_DIR
ls -ld "$SCRIPT_DIR"
mkdir -p ${SCRIPT_DIR}/output  
mkdir -p ${SCRIPT_DIR}/ckpts


NUM_GPUS=8
CONTAINER_NAME=${train_imseg}
BATCH_SIZE=2
DOCKER_MEMORY=${4:-""}

DOCKER_MEMORY_PARAM=

if [ ! -z "$DOCKER_MEMORY" ]
then
	DOCKER_MEMORY_PARAM="-m ${DOCKER_MEMORY}g"
fi

docker run --ipc=host \
           --gpus all \
           --privileged \
            -u $(id -u rnouaj):$(id -g rnouaj) \
           --volume /run/udev:/run/udev:ro \
           --volume .:/workspace \
           --volume /sys/kernel/config:/sys/kernel/config:ro \
           --volume /dev:/dev:ro \
           -v /usr/local/cuda12.5:/mnt \
           --name="train_imseg" \
           --runtime=nvidia \
           -it --rm $DOCKER_MEMORY_PARAM \
           -v /raid/data/imseg/raw-data/kits19/data/:/raw_data \
            -v /raid/data/imseg/raw-data/kits19/200gb_data/:/data \
           -v ${SCRIPT_DIR}/output:/results \
           -v ${SCRIPT_DIR}/ckpts:/ckpts \
           unet3d:dali bash
