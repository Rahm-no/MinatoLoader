docker run --ipc=host --name=train_imseg -it --rm --runtime=nvidia \
    -v /raid/data/imseg/raw-data/kits19/data:/raw_data \
    -v /raid/data/imseg/raw-data/kits19/preproc-data:/data \
    -v $(pwd)/results:/results \
    -v $(pwd)/ckpts:/ckpts \
    -v $(pwd):/workspace/unet3d \
    minato:latest


# docker run --ipc=host --name=train_imseg -it --rm --runtime=nvidia \
#     -v  RAW-DATA-DIR:/raw_data \
#     -v PREPROCESSED-DATA-DIR:/data \
#     -v $(pwd)/results:/results \
#     -v $(pwd)/ckpts:/ckpts \
#     -v $(pwd):/workspace/unet3d \
#     minato:latest


