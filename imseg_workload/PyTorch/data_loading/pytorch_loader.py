import random
import numpy as np
import scipy.ndimage
from torch.utils.data import Dataset
from torchvision import transforms
import os
from runtime.logging import mllog_event

def get_train_transforms():

    rand_flip = RandFlip()
    cast = Cast(types=(np.float32, np.uint8))
    rand_scale = RandomBrightnessAugmentation(factor=0.3, prob=1)
    rand_noise = GaussianNoise(mean=0.0, std=0.1, prob=1)
    heavy_blur = HeavyGaussianBlur(sigma_range=(1, 5))
    elastic_deform = ElasticDeformation(alpha=1, sigma=5)
    rand_noise = transforms.Compose([rand_noise])
    train_transforms = transforms.Compose([
        rand_flip, cast, rand_scale, rand_noise, 
    ])
    return train_transforms


class HeavyGaussianBlur:
    def __init__(self, sigma_range=(1, 5)):
        self.sigma_range = sigma_range

    def __call__(self, data):
        sigma = np.random.uniform(*self.sigma_range)
        image = data["image"]
        for c in range(image.shape[0]):
            image[c] = scipy.ndimage.gaussian_filter(image[c], sigma=sigma)
        data["image"] = image
        return data


class ElasticDeformation:
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, data):
        image = data["image"]
        shape = image.shape[1:]

        # Generate random displacement fields
        dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dz = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha

        # Create meshgrid
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = (x + dx).reshape(-1), (y + dy).reshape(-1), (z + dz).reshape(-1)

        # Apply deformation
        for c in range(image.shape[0]):
            image[c] = scipy.ndimage.map_coordinates(image[c], indices, order=1, mode='reflect').reshape(shape)

        data["image"] = image
        return data


class RandBalancedCrop:
    def __init__(self, patch_size, oversampling):
        self.patch_size = patch_size
        self.oversampling = oversampling

    def __call__(self, data):
        image, label = data["image"], data["label"]
        for i in range(5):
            image_new, label_new, cords_new = self.rand_foreg_cropd(image, label)
        
            image1, label1, cords1 = self._rand_crop(image, label)
        # print("image_new.shape", image_new.shape)
        data.update({"image": image_new, "label": label_new})
        return data

    @staticmethod
    def randrange(max_range):
        return 0 if max_range == 0 else random.randrange(max_range)

    def get_cords(self, cord, idx):
        return cord[idx], cord[idx] + self.patch_size[idx]

    def _rand_crop(self, image, label):
        ranges = [s - p for s, p in zip(image.shape[1:], self.patch_size)]
        # print("ranges", ranges)
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
        self.prob = 1 

    def flip(self, data, axis):
        data["image"] = np.flip(data["image"], axis=axis).copy()
        data["label"] = np.flip(data["label"], axis=axis).copy()
        return data

    def __call__(self, data):
        for axis in self.axis:
            # if random.random() < self.prob:
            data = self.flip(data, axis)
        return data


class Cast:
    def __init__(self, types):
        self.types = types

    def __call__(self, data):
        # print("data['image'].dtype before cast", data["image"].dtype)
        # print("data['label'].dtype before cast", data["label"].dtype)
        # data["image"] = data["image"].astype(self.types[0])
        data["label"] = data["label"].astype(self.types[1])
        # print("data['image'].dtype after cast", data["image"].dtype)
        # print("data['label'].dtype after cast", data["label"].dtype)
        return data


class RandomBrightnessAugmentation:
    def __init__(self, factor, prob):
        self.prob = prob
        self.factor = factor

    def __call__(self, data):
        image = data["image"]
        for i in range(1):
            factor = np.random.uniform(low=1.0-self.factor, high=1.0+self.factor, size=1)
            image = (image * (1 + factor)).astype(image.dtype)
            data.update({"image": image})
        return data


class GaussianNoise:
    def __init__(self, mean, std, prob):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, data):
        image = data["image"]
        for i in range(1):
            scale = np.random.uniform(low=0.0, high=self.std)
            noise = np.random.normal(loc=self.mean, scale=scale, size=image.shape).astype(image.dtype)
            data.update({"image": image + noise})
        return data


class PytTrain(Dataset):
    def __init__(self, images, labels, **kwargs):
        self.images, self.labels = images, labels
        self.train_transforms = get_train_transforms()
        patch_size, oversampling = kwargs["patch_size"], kwargs["oversampling"]
        self.patch_size = patch_size
        self.rand_crop = RandBalancedCrop(patch_size=patch_size, oversampling=oversampling)
        prep_time = 0 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = {"image": np.load(self.images[idx]), "label": np.load(self.labels[idx])}
        data = self.rand_crop(data)
        data = self.train_transforms(data)
 

        return data["image"], data["label"]


class PytVal(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return np.load(self.images[idx]), np.load(self.labels[idx])






