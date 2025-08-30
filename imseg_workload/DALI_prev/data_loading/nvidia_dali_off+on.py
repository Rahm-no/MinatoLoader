import numpy as np
import pandas as pd
import scipy.ndimage
from torch.utils.data import Dataset
from torchvision import transforms
import os
from runtime.logging import mllog_event,get_dllogger, mllog_start, mllog_end, mllog_event, mlperf_submission_log, mlperf_run_param_log
from runtime.arguments import PARSER
import glob
import time
import logging
import random
import subprocess
import argparse
import numpy as np
import torch
from torch.nn.functional import interpolate
from memory_profiler import profile
from preprocess_dataset import Preprocessor
import concurrent.futures
import subprocess
import io
import nibabel
import cProfile
import pstats

#-----------------Trying to combine offline and online preprocessing----------------------------

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
        #mllog_start(key='previous Offline  preprocessing starts for',value="Case {}".format(case))

        image, label = self.preprocess_case(image, label, image_spacings,case)
        image, label = self.pad_to_min_shape(image, label,case)
        data.update({"image": image, "label": label})
        #mllog_end(key='previous Offline  preprocessing ends for',value="Case {}".format(case))


        
        return data



    def preprocess_case(self, image, label, image_spacings, case):
        
        image, label = self.resample3d(image, label, image_spacings,case)
        image = self.normalize_intensity(image.copy(),case)
        return image, label

    @staticmethod
    def pad_to_min_shape(image, label,case):
        #mllog_start(key='Min padding_start',value="Case {}".format(case))
 
        current_shape = image.shape[1:]
        #mllog_event(key="image shape Case {}".format(case),value=image.shape)

        bounds = [max(0, TARGET_SHAPE[i] - current_shape[i]) for i in range(3)]
        paddings = [(0, 0)]
        paddings.extend([(bounds[i] // 2, bounds[i] - bounds[i] // 2) for i in range(3)])
        
        x=np.pad(image, paddings, mode="edge"), np.pad(label, paddings, mode="edge")
        #mllog_event(key="image shape Case {}".format(case),value=x[0].shape)


        #mllog_end(key='Min padding_end',value="Case {}".format(case))

        return x


    def resample3d(self, image, label, image_spacings, case):
        #mllog_start(key='3DResampling_start',value="Case {}".format(case))
        #mllog_event(key="image shape Case {}".format(case),value=image.shape)

      
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

        #mllog_event(key="image shape Case {}".format(case),value=image.shape)



 
        #mllog_end(key='3DResampling_end',value="Case {}".format(case))

        
        return image, label

    def normalize_intensity(self, image: np.array, case: str):
        #mllog_start(key='NormalizeIntensity_start',value="Case {}".format(case))
        #mllog_event(key="image shape Case {}".format(case),value=image.shape)
     
        image = np.clip(image, self.min_val, self.max_val)
        image = (image - self.mean) / self.std
    
        #mllog_end(key='NormalizeIntensity_end',value="Case {}".format(case))

        return image
    
  #--------------------------------------------------------------------------------------------




def get_train_transforms():
    rand_flip = RandFlip()
    cast = Cast(types=(np.float32, np.uint8))
    rand_scale = RandomBrightnessAugmentation(factor=0.3, prob=0.1)
    rand_noise = GaussianNoise(mean=0.0, std=0.1, prob=0.1)
    train_transforms = transforms.Compose([rand_flip, cast, rand_scale, rand_noise])
    return train_transforms




class RandBalancedCrop:
    def __init__(self, patch_size, oversampling):
        self.patch_size = patch_size
        self.oversampling = oversampling
   # @profile
    def __call__(self, data):
        #mllog_start(key='RandCrop_start', value=data["image_name"], sync=True)
        

        image, label = data["image"], data["label"]
        #mllog_event(key='RandCrop starts Case image  {} shape'.format(data['image_name']), value = image.shape,sync=True )
        #profiler = cProfile.Profile()
        #profiler.enable()

        #if random.random() < self.oversampling:
         #   image, label, cords = self.rand_foreg_cropd(image, label)
        #else:
        image, label, cords = self._rand_crop(image, label)

        data.update({"image": image, "label": label})
        #profiler.disable()
        #profiler_stats = pstats.Stats(profiler) 
        #cpu_time = profiler_stats.total_tt
        #print('this is the time spent by the cpu', cpu_time)
        #profiler_stats.sort_stats(pstats.SortKey.CUMULATIVE)
        #profiler_stats.print_stats()
        
        

        #mllog_end(key='RandCrop_end', value=data["image_name"], sync=True)
        #mllog_event(key='Rand crop ends Case image  {} shape'.format(data['image_name']), value = image.shape,sync=True )

        return data


    @staticmethod
    def randrange(max_range):
        return 0 if max_range == 0 else random.randrange(max_range)

    def get_cords(self, cord, idx):
        return cord[idx], cord[idx] + self.patch_size[idx]

    def _rand_crop(self, image, label):
        ranges = [s - p for s, p in zip(image.shape[1:], self.patch_size)]
        #mllog_event(key='image shape for ranges and patch size',value=(image.shape[1:], self.patch_size ))
        cord = [self.randrange(x) for x in ranges]
        low_x, high_x = self.get_cords(cord, 0)
        low_y, high_y = self.get_cords(cord, 1)
        low_z, high_z = self.get_cords(cord, 2)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]

    def rand_foreg_cropd(self, image, label):
        def adjust(foreg_slice, patch_size, label, idx):
            diff = patch_size[idx - 1] - (foreg_slice[idx].stop - foreg_slice[idx].start)
            sign = -1 if diff < 0 else 1
            diff = abs(diff)
            ladj = self.randrange(diff)
            hadj = diff - ladj
            low = max(0, foreg_slice[idx].start - sign * ladj)
            high = min(label.shape[idx], foreg_slice[idx].stop + sign * hadj)
            diff = patch_size[idx - 1] - (high - low)
            if diff > 0 and low == 0:
                high += diff
            elif diff > 0:
                low -= diff
            return low, high

        cl = np.random.choice(np.unique(label[label > 0]))
        foreg_slices = scipy.ndimage.find_objects(scipy.ndimage.measurements.label(label==cl)[0])
        foreg_slices = [x for x in foreg_slices if x is not None]
        slice_volumes = [np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices]
        slice_idx = np.argsort(slice_volumes)[-2:]
        foreg_slices = [foreg_slices[i] for i in slice_idx]
        if not foreg_slices:
            return self._rand_crop(image, label)
        foreg_slice = foreg_slices[random.randrange(len(foreg_slices))]
        low_x, high_x = adjust(foreg_slice, self.patch_size, label, 1)
        low_y, high_y = adjust(foreg_slice, self.patch_size, label, 2)
        low_z, high_z = adjust(foreg_slice, self.patch_size, label, 3)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]



class RandFlip:
    def __init__(self):
        self.axis = [1, 2, 3]
        self.prob = 1 / len(self.axis)

    def flip(self, data, axis):
        data["image"] = np.flip(data["image"], axis=axis).copy()
        data["label"] = np.flip(data["label"], axis=axis).copy()
        return data

    def __call__(self, data):
        #mllog_start(key='RandFlip_start', value=data["image_name"], sync=True)        
        for axis in self.axis:
            #if random.random() < self.prob:
                data = self.flip(data, axis)
        #mllog_end(key='RandFlip_end', value=data["image_name"], sync=True)  
        return data






class Cast:
    def __init__(self, types):
        self.types = types
    #@profile
    def __call__(self, data):
        #mllog_start(key='Cast_start', value= data["image_name"],  sync=True)        
        #mllog_event(key='Case image  {} shape'.format(data['image_name']), value = data['image'].shape,sync=True )

      #  profiler = cProfile.Profile()
      #  profiler.enable()



        data["image"] = data["image"].astype(self.types[0])
        data["label"] = data["label"].astype(self.types[1])

      #  profiler.disable()
      #  # Display the profiler output
      #  profiler_stats = pstats.Stats(profiler)
       # cpu_time = profiler_stats.total_tt
       # print('this is the time spent by the cpu', cpu_time)
       # profiler_stats.sort_stats(pstats.SortKey.CUMULATIVE)
       # profiler_stats.print_stats()
        #mllog_event(key='Case image  {} shape'.format(data['image_name']), value = data['image'].shape,sync=True )

        #mllog_end(key='Cast_end', value= data["image_name"],sync=True)
        return data





class RandomBrightnessAugmentation:
    def __init__(self, factor, prob):
        self.prob = prob
        self.factor = factor
    #@profile
    def __call__(self, data):
        #perf_command = f' perf stat -a sleep 5'

        
        #mllog_start(key='RandomBrightnessAugmentation_start', value=data["image_name"], sync=True)
      #  profiler = cProfile.Profile()
     #   profiler.enable()
        image = data["image"]


        #if random.random() < self.prob:
        factor = np.random.uniform(low=1.0 - self.factor, high=1.0 + self.factor, size=1)
        image = (image * (1 + factor)).astype(image.dtype)
        data.update({"image": image})
       # profiler.disable()
        # Display the profiler output
        #profiler_stats = pstats.Stats(profiler)
        #cpu_time = profiler_stats.total_tt
        #print('this is the time spent by the cpu', cpu_time)
        #profiler_stats.sort_stats(pstats.SortKey.CUMULATIVE)
        #profiler_stats.print_stats()
        #mllog_event(key='Case image  {} shape'.format(data['image_name']), value = image.shape,sync=True )
        

        #mllog_end(key='RandomBrightnessAugmentation_end', value=data["image_name"], sync=True)
        
        return data



class GaussianNoise:
    def __init__(self, mean, std, prob):
        self.mean = mean
        self.std = std
        self.prob = prob
        
 #   @profile(precision = 4)
    def __call__(self, data):
        #mllog_start(key='RandGaussianNoise_start', value= data["image_name"],sync=True)
      
        image = data["image"]
        #mllog_event(key='Case image shape', value = image.shape,sync=True )
  #      profiler = cProfile.Profile()
   #     profiler.enable()
        #if random.random() < self.prob:
        scale = np.random.uniform(low=0.0, high=self.std)
        noise = np.random.normal(loc=self.mean, scale=scale, size=image.shape).astype(image.dtype)
        data.update({"image": image + noise})
        

       
    #    profiler.disable()
    #    profiler_stats = pstats.Stats(profiler)
    #    cpu_time = profiler_stats.total_tt
     #   print('this is the time spent by the cpu', cpu_time)
     #   profiler_stats.sort_stats(pstats.SortKey.CUMULATIVE)
     #   profiler_stats.print_stats()
        #mllog_end(key='RandGaussianNoise_end', value= data["image_name"], sync=True)

        
      

        return data




class ImageSpacingExtractor:
    def __init__(self, data_root):
        self.data_root = data_root

    def get_image_spacing(self, case):
        image_path = os.path.join(self.data_root, case, "imaging.nii.gz")

        if os.path.exists(image_path):
            header = nibabel.load(image_path).header
            image_spacing = header["pixdim"][1:4].tolist()
            return image_spacing
        else:
            return None  # Image file does not exist

class preprocessing:

    def __init__(self,batch_size,batch):
        self.batch = batch
        self.batch_size = batch_size
        #self.offline_prep = offline_preprocessing()
       # patch_size, oversampling = kwargs["patch_size"], kwargs["oversampling"]
       # self.rand_crop = RandBalancedCrop(patch_size=patch_size, oversampling=oversampling)


    def get_train_transforms(self):
    #------------------------------------------------------------------------------------------
  # The only functions that I could split them from the dataloader
    #------------------------------------------------------------------------------------------
        
        rand_flip = RandFlip()
        
        cast = Cast(types=(np.float32, np.uint8))
        rand_scale = RandomBrightnessAugmentation(factor=0.3, prob=0.1)
    

        rand_noise = GaussianNoise(mean=0.0, std=0.1, prob=0.1)


        train_transforms = transforms.Compose([rand_flip, cast, rand_scale, rand_noise])
        return train_transforms

    def __call__(self):
        _,_,data = self.batch
        train_transforms =  self.get_train_transforms()  # Initialize train_transforms inside __call__
        #print('image shape before randomflip from data dataloader',data["image"][0].shape)
        
       
        
        for i in range(self.batch_size): 
            
            prepdata = {
            "case": data["case"][i],
            "image": data["image"][i],
            "label": data["label"][i],
            "image_name": data["image_name"][i],
            "label_name": data["label_name"][i]
            
            }

            
            #print('imageshape from prepdata',prepdata['image'].shape)
            #print('prepdata',prepdata)
           


            prepdata =  train_transforms(prepdata)
            data["case"][i] = prepdata["case"]
            data["image"][i] = torch.from_numpy(prepdata["image"])
            data["label"][i] = torch.from_numpy(prepdata["label"])
            data["image_name"][i] = prepdata["image_name"]
            data["label_name"][i] = prepdata["label_name"]

        mllog_event(key='data inside preprocessing class', value = data, sync= True)

            

       
        
        return data
    
            
        

class PytTrain(Dataset):
   

    def __init__(self, images, labels, **kwargs):
        self.images, self.labels = images, labels
        self.offline_prep = offline_preprocessing()
        self.train_transforms = get_train_transforms()
        patch_size, oversampling = kwargs["patch_size"], kwargs["oversampling"]
        raw_dir = kwargs["raw-dir"]  # Fix the key here
        image_spacings = kwargs["image_spacings"]
        self.patch_size = patch_size
        self.raw_dir = raw_dir
        self.image_sp_dict = image_spacings
        self.rand_crop = RandBalancedCrop(patch_size=patch_size, oversampling=oversampling)

   
    
    def __len__(self):
        return len(self.images)
    

    
        
   
   
    def __getitem__(self, idx):
           # print("idx",idx)
       
     
            image_path = self.images[idx]
            label_path = self.labels[idx]
            #print("Loading:", image_path, label_path)



            image_name = os.path.basename(image_path)
            label_name = os.path.basename(label_path)
            name_without_extension = image_name.split('.')[0]  # Split by period and take the first part
            case_exp = name_without_extension.split('_')  # Split by underscore

            case = case_exp[0] +'_' + case_exp[1]

        
            image_spacings_case =   self.image_sp_dict[case]
        

            data = {
                "case": case,
                "image": np.load(image_path),
                "label": np.load(label_path),
                "image_name": image_name,
                "label_name": label_name,
                "image_spacings" : image_spacings_case}
            


            
        
            mllog_start(key='Online preprocessing starts')

            data = self.offline_prep(data)
            data = self.rand_crop(data)

            data = self.train_transforms(data)
            #subprocess.run(['nvidia-smi'])
            mllog_end(key='Online preprocessing ends')
            
       



            return data["image"], data["label"]
        
        
        
        

        # Display GPU utilization using nvidia-smi
        #subprocess.run(['nvidia-smi'])
        
        #return data["image"], data["label"], data
      



    





class PytVal(Dataset):
    def __init__(self, images, labels,image_spacings):
        self.images, self.labels = images, labels
        self.offline_prep = offline_preprocessing()
        self.image_sp_dict= image_spacings



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]
        label_path = self.labels[idx]
        #print("Loading:", image_path, label_path)
        


        image_name = os.path.basename(image_path)
        label_name = os.path.basename(label_path)
        name_without_extension = image_name.split('.')[0]  # Split by period and take the first part
        case_exp = name_without_extension.split('_')  # Split by underscore

        case = case_exp[0] +'_' + case_exp[1]

    
        image_spacings_case =   self.image_sp_dict[case]
      

        data = {
            "case": case,
            "image": np.load(image_path),
            "label": np.load(label_path),
            "image_name": image_name,
            "label_name": label_name,
            "image_spacings" : image_spacings_case}

        
        data = self.offline_prep(data)
        return data["image"],data["label"]




