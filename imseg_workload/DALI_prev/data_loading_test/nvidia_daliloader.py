import os
import glob
from random import shuffle
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import shutil
from data_loading.pytorch_loader import PytVal, PytTrain
from runtime.logging import mllog_event
import numpy as np
import nvidia.dali.fn as fn
import random

import random
import numpy as np
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline import pipeline_def




MEAN_VAL = 101.0
STDDEV_VAL = 76.9
MIN_CLIP_VAL = -79.0
MAX_CLIP_VAL = 304.0
TARGET_SPACING = [1.6, 1.2, 1.2]
TARGET_SHAPE = [128, 128, 128]

class offline_preprocessing:
    def __init__(self):
        self.mean = MEAN_VAL
        self.std = STDDEV_VAL
        self.min_val = MIN_CLIP_VAL
        self.max_val = MAX_CLIP_VAL
        self.target_spacing = TARGET_SPACING
    def __call__(self, data):
       
        image , label , image_spacings, case = data["image"], data["label"], data["image_spacings"], data["case"]

        image, label = self.preprocess_case(image, label, image_spacings,case)
        image, label = self.pad_to_min_shape(image, label,case)
        data.update({"image": image, "label": label})


        
        return data



    def preprocess_case(self, image, label, image_spacings, case):
        
        image, label = self.resample3d(image, label, image_spacings,case)
        image = self.normalize_intensity(image.copy(),case)
        return image, label

    @staticmethod
    def pad_to_min_shape(image, label,case):
 
        current_shape = image.shape[1:]

        bounds = [max(0, TARGET_SHAPE[i] - current_shape[i]) for i in range(3)]
        paddings = [(0, 0)]
        paddings.extend([(bounds[i] // 2, bounds[i] - bounds[i] // 2) for i in range(3)])
        
        x=np.pad(image, paddings, mode="edge"), np.pad(label, paddings, mode="edge")

        return x


    def resample3d(self, image, label, image_spacings, case):
       
        if image_spacings != self.target_spacing:
            spc_arr = np.array(image_spacings)
            targ_arr = np.array(self.target_spacing)
            shp_arr = np.array(image.shape[1:])
            new_shape = (spc_arr / targ_arr * shp_arr).astype(int).tolist()

            image = interpolate(torch.from_numpy(np.expand_dims(image, 0)),
                                size=new_shape, mode='trilinear', align_corners=True)
            label = interpolate(torch.from_numpy(np.expand_dims(label, 0)), size=new_shape, mode='nearest')
            image = np.squeeze(image.numpy(), 0)
            label = np.squeeze(label.numpy(), 0)

    
        
        return image, label

    def normalize_intensity(self, image: np.array, case: str):
    
     
        image = np.clip(image, self.min_val, self.max_val)
        image = (image - self.mean) / self.std
    

        return image
    
  #--------------------------------------------------------------------------------------------




def list_files_with_pattern(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data



def get_split(data, train_idx, val_idx):
    train = list(np.array(data)[train_idx])
    val = list(np.array(data)[val_idx])
    return train, val


import os
import numpy as np

def save_data(path, data):
    print(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)
    # Get the directory paths
    
    

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
    
    # imgs_val, lbls_val = split_eval_data(imgs_val, lbls_val, num_shards, shard_id)


    return imgs_train, lbls_train, imgs_val, lbls_val


from nvidia.dali import pipeline_def, fn, types
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator


import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import torch
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import random
import numpy as np
from nvidia.dali.pipeline import Pipeline

# Custom transformation to handle RandBalancedCrop (can be complex, might need custom operator)
def rand_balanced_crop(image, label, patch_size, oversampling):
    # Similar to your RandBalancedCrop logic
    
        # Apply foreground cropping (similar logic as in _rand_foreg_cropd)
    return fn.crop(image, crop=(patch_size[0], patch_size[1], patch_size[2]),out_of_bounds_policy="pad")

 
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types


def sharded_pipeline(files1, files2, device_id, shard_id, num_shards, patch_size, oversampling, batch_size, training=True):
  
    print("batch_size in pipeline " , batch_size, "shard_id", shard_id, "num_shards", num_shards)


 # Modified pipeline configuration
    pipe = Pipeline(
        batch_size=batch_size,
        num_threads=2,  # Increased to match CPU cores
        device_id=device_id ,       
        prefetch_queue_depth=8,  # Increased prefetch depth
    )

    with pipe:
        # Read + load
        images = fn.readers.numpy(device="cpu", files=files1, shard_id=shard_id, num_shards=num_shards, name="reader")
        labels = fn.readers.numpy(device="cpu", files=files2, shard_id=shard_id, num_shards=num_shards, name="reader_labels")
        images = images.gpu()
        labels = labels.gpu()

        # Normalize types
        images = fn.cast(images, dtype=types.FLOAT, device="gpu")
        labels = fn.cast(labels, dtype=types.UINT8, device="gpu")

        # Change layout to (H, W, D, C)
        images = fn.transpose(images, perm=[1, 2, 3, 0], device="gpu")
        labels = fn.transpose(labels, perm=[1, 2, 3, 0], device="gpu")

        # Spatial crop
        images = fn.crop(images, crop=patch_size, device="gpu")
        labels = fn.crop(labels, crop=patch_size, device="gpu")

        # Flip: random axes [x, y, z]
        # for i in range(50):
        flip_axes = fn.random.coin_flip(probability=0.5, shape=3)
        images = fn.flip(images, horizontal=flip_axes[0], vertical=flip_axes[1], depthwise=flip_axes[2])
        labels = fn.flip(labels, horizontal=flip_axes[0], vertical=flip_axes[1], depthwise=flip_axes[2])

        # Brightness & Contrast
        images = fn.brightness_contrast(
            images,
            brightness=fn.random.uniform(range=[0.8, 1.2]),
            contrast=fn.random.uniform(range=[0.8, 1.2]),
            contrast_center=128
        )

        # Gaussian noise + blur
        images = fn.gaussian_blur(
            images + fn.random.normal(images, stddev=0.05),
            window_size=21,
            sigma=fn.random.uniform(range=[0.5, 1.5])
        )

        angle = fn.random.uniform(range=[-10.0, 10.0])  # small 3D rotation
        matrix = fn.transforms.rotation(angle=angle, axis=[0.0, 0.0, 1.0])  # rotate in XY
        images = images.gpu()
        labels = labels.gpu()
        images = fn.warp_affine(images, matrix=matrix, device="gpu", fill_value=0)
        labels = fn.warp_affine(labels, matrix=matrix, device="gpu", fill_value=0, interp_type=types.INTERP_NN)

        # Back to (C, D, H, W)
        images = fn.transpose(images, perm=[3, 0, 1, 2], device="gpu")
        labels = fn.transpose(labels, perm=[3, 0, 1, 2], device="gpu")


        pipe.set_outputs(images, labels)

    return pipe

def get_data_loaders(flags, num_shards, global_rank):
    patch_size = flags.input_shape
    oversampling = flags.oversampling
    # Assuming get_data_split is a function that returns training and validation data files

    x_train, y_train, x_val, y_val = get_data_split(flags.data_dir, num_shards, shard_id=global_rank)
    print("len(x_va)", len(x_val))
    print("len(y_val)", len(y_val))

    # Create DALI pipelines for training and validation
    train_pipe = sharded_pipeline(x_train, y_train, device_id= global_rank, shard_id=global_rank, num_shards=num_shards,
                                  patch_size=patch_size, oversampling=oversampling,batch_size=flags.batch_size, training=True)
    val_pipe = sharded_pipeline(x_val, y_val, device_id= global_rank, shard_id=global_rank, num_shards=num_shards, patch_size=patch_size, 
                                oversampling=oversampling,batch_size =1, training=False)  # Validation uses single GPU

    # Build pipelines
    train_pipe.build()
    val_pipe.build()

        
    output_map = ["data", "label"]  # Must match your pipeline output order


    # Modified iterator configuration
    dali_train_iter = DALIGenericIterator(
        pipelines=train_pipe,
        size=-1,  # Let DALI auto-calculate size
        output_map=output_map,
        reader_name="reader",  # Important for sharding
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        prepare_first_batch=True
    )

    dali_val_iter = DALIGenericIterator(
        pipelines=val_pipe,
        size = 42 ,
        output_map=output_map,
        reader_name=None,
        auto_reset=True,)
    

    print("train_iter", dali_train_iter)
    return dali_train_iter, dali_val_iter