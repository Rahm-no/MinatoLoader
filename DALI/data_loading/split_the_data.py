import os
import glob
from random import shuffle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import shutil
from data_loading.pytorch_loader import PytVal, PytTrain
from runtime.logging import mllog_event
import numpy as np


import random
import numpy as np



def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data






import os
import numpy as np

def save_data(path, data):
    print(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)
    
    

def split_eval_data(x_val, y_val, num_shards, shard_id):
    x = [a.tolist() for a in np.array_split(x_val, num_shards)]
    y = [a.tolist() for a in np.array_split(y_val, num_shards)]
    return x[shard_id], y[shard_id]

def get_data_split(path: str, num_shards: int, shard_id: int):
    with open("evaluation_cases.txt", "r") as f:
        val_cases_list = f.readlines()
    val_cases_list = [case.rstrip("\n") for case in val_cases_list]
    
    # Load image and label files
    imgs = load_data(path, "*_x.npy")
    lbls = load_data(path, "*_y.npy")
    
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    
    imgs_train, lbls_train, imgs_val, lbls_val = [], [], [], []
    
    for (case_img, case_lbl) in zip(imgs, lbls):
        if case_img.split("_")[-2] in val_cases_list:
            imgs_val.append(case_img)
            lbls_val.append(case_lbl)
        else:
            imgs_train.append(case_img)
            lbls_train.append(case_lbl)
    
    mllog_event(key='train_samples', value=len(imgs_train), sync=False)
    mllog_event(key='eval_samples', value=len(imgs_val), sync=False)
    
    imgs_val, lbls_val = split_eval_data(imgs_val, lbls_val, num_shards, shard_id)

    for e in imgs_val:
        filepath =  os.path.join(path,"data_split", "val_images", os.path.basename(e))
        save_data(os.path.join(path, "val_images", os.path.basename(e)), e)
        print(f"{file_path}: {data.dtype}")

    for e in lbls_val: 
         save_data(os.path.join(path,"data_split", "val_labels", os.path.basename(e)), e)
    for e in imgs_train: 
        
        save_data(os.path.join(path,"data_split", "train_images", os.path.basename(e)), e)
    for e in lbls_train: 
        save_data(os.path.join(path, "data_split","train_labels", os.path.basename(e)), e)
    train_images_dir = os.path.join(path, "train_images")
    num_train_images = len(os.listdir(train_images_dir))
    print(f"Number of train images: {num_train_images}")
    print("size of the list", len (imgs_train))
    


   

if __name__==main:
    get_data_split("/raid/data/unet3d/rawdata_npy",)




