#!/bin/bash

if [[ ! -d results ]]
then
    mkdir results
fi

if [[ ! -d ckpts ]]
then
    mkdir ckpts
fi

docker run --ipc=host --name=train_imseg -it --rm --runtime=nvidia -v /data/kits19/data/:/raw_data -v /data/kits19/preprocessed_data/:/data -v $(pwd)/results:/results -v $(pwd)/ckpts:/ckpts unet3d:latest /bin/bash
