#!/usr/bin/env bash
set -e

echo "Fixing permissions..."
find . -type f -name "*.sh" -exec chmod +x {} \;
echo "Done "

set -euo pipefail

# docker run --ipc=host --name=train_imseg -it --rm --runtime=nvidia \
#     -v /raid/data/imseg/raw-data/kits19/data:/raw_data \
#     -v /raid/data/imseg/raw-data/kits19/preproc-data:/data \
#     -v $(pwd)/results:/results \
#     -v $(pwd)/ckpts:/ckpts \
#     -v $(pwd):/workspace/unet3d \
#     minato:latest

# Resolve host directories to absolute paths
RAW_DATA=$(realpath raw-data-dir/kits19/data)
PREPROC_DATA=$(realpath raw-data-dir/kits19/preproc-data)
RESULTS=$(realpath ./results)
CKPTS=$(realpath ./ckpts)
WORKDIR=$(realpath .)

CONTAINER_NAME=${CONTAINER_NAME:-train_imseg}
IMAGE=${IMAGE:-minato:latest}

echo "Using paths:"
echo "  RAW_DATA     = $RAW_DATA"
echo "  PREPROC_DATA = $PREPROC_DATA"
echo "  RESULTS      = $RESULTS"
echo "  CKPTS        = $CKPTS"
echo "  WORKDIR      = $WORKDIR"
echo "  IMAGE        = $IMAGE"
echo "  CONTAINER    = $CONTAINER_NAME"

docker run --ipc=host --name "$CONTAINER_NAME" -it --rm --gpus all \
    -v "$RAW_DATA":/raw_data \
    -v "$PREPROC_DATA":/data \
    -v "$RESULTS":/results \
    -v "$CKPTS":/ckpts \
    -v "$WORKDIR":/workspace/unet3d \
    "$IMAGE"
