from collections import namedtuple

import cv2
import numpy as np
import pandas as pd
from collections import defaultdict

import math

from scipy import ndimage
from torch.utils.data import Dataset
import skimage.color
import skimage.io
from tqdm import tqdm
import utils
import glob
import pickle
import enum

import skimage.transform

import matplotlib.pyplot as plt

from config import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class TransformCfg:
    """
    Configuration structure for crop parameters.
    """

    def __init__(self,
                 crop_size,
                 src_center_x,
                 src_center_y,
                 scale_x=1.0,
                 scale_y=1.0,
                 angle=0.0,
                 shear=0.0,
                 hflip=False,
                 vflip=False):
        self.crop_size = crop_size
        self.src_center_x = src_center_x
        self.src_center_y = src_center_y
        self.angle = angle
        self.shear = shear
        self.scale_y = scale_y
        self.scale_x = scale_x
        self.vflip = vflip
        self.hflip = hflip

    def __str__(self):
        return str(self.__dict__)

    def transform(self):
        scale_x = self.scale_x
        if self.hflip:
            scale_x *= -1
        scale_y = self.scale_y
        if self.vflip:
            scale_y *= -1

        tform = skimage.transform.AffineTransform(translation=(self.src_center_x, self.src_center_y))
        tform = skimage.transform.AffineTransform(scale=(1.0 / self.scale_x, 1.0 / self.scale_y)) + tform
        tform = skimage.transform.AffineTransform(rotation=self.angle * math.pi / 180,
                                                  shear=self.shear * math.pi / 180) + tform
        tform = skimage.transform.AffineTransform(translation=(-self.crop_size / 2, -self.crop_size / 2)) + tform

        return tform

    def transform_image(self, img):
        crop = skimage.transform.warp(img, self.transform(),
                                      mode='constant',
                                      cval=0,
                                      order=1,
                                      output_shape=(self.crop_size, self.crop_size))
        # crop = np.clip(crop, 0, 255).astype(np.uint8)
        return crop


TrainingSample = namedtuple('TrainingSample', 'image_id labels is_external')
GeometryAugLevel = namedtuple('GeometryAugLevel', 'prob scale scale_xy angle shear')
ImageAugLevel = namedtuple('ImageAugLevel', 'brightness_common brightness gamma_common gamma mul_noise mul_noise_p '
                                            'blur_level blur_p')


class ClassificationDataset(Dataset):
    def __init__(self, fold, is_training,
                 transform=None,
                 geometry_aug_level=10,
                 img_aug_level=20,
                 crop_size=512,
                 folds_split='emb',
                 use_extarnal=False):
        self.geometry_aug_level = geometry_aug_level
        self.img_aug_level = img_aug_level
        self.fold = fold
        self.is_training = is_training
        self.transform = transform
        self.images = {}
        self.crop_size = crop_size

        self.samples = []
        self.samples_idx_by_label = defaultdict(list)

        if folds_split == 'orig':
            print('using orig folds split')
            data = pd.read_csv('../input/folds_4.csv')
        elif folds_split == 'emb':
            print('using emb folds split')
            data = pd.read_csv('../input/folds_4_emb.csv')
        elif folds_split == 'cluster4x':
            print('using cluster4x folds split')
            clusters = pd.read_csv('../input/cluster4x_folds.csv')
            train = pd.read_csv('../input/train.csv')
            data = pd.merge(train, clusters, on='Id', how='left')
            data['fold'] = data['cluster4']
        # data = pd.read_csv('../input/folds_4.csv')
        # data = pd.read_csv('../input/folds_4_emb.csv')
        for _, row in data.iterrows():
            if is_training == (fold != row['fold']):
                lbl = [int(l) for l in row['Target'].split(' ')]
                labels = np.zeros(NB_CATEGORIES, dtype=np.float32)
                labels[lbl] = 1.0
                sample = TrainingSample(image_id=row['Id'], labels=labels, is_external=False)

                self.samples.append(sample)
                for key in lbl:
                    self.samples_idx_by_label[key].append(len(self.samples) - 1)

        if use_extarnal and is_training:
            data = pd.read_csv('../input/folds_4_extra.csv')
            for _, row in data.iterrows():
                lbl = [int(l) for l in row['Target'].split(' ')]
                labels = np.zeros(NB_CATEGORIES, dtype=np.float32)
                labels[lbl] = 1.0
                sample = TrainingSample(image_id=row['Id'], labels=labels, is_external=True)

                self.samples.append(sample)
                for key in lbl:
                    self.samples_idx_by_label[key].append(len(self.samples) - 1)

    def load_images(self, sample):
        image_id = sample.image_id
        if image_id not in self.images:
            images = []
            # train_dir = TRAIN_DIR_1024 if self.crop_size == 1024 else TRAIN_DIR
            train_dir = {
                512: {False: TRAIN_DIR, True: TRAIN_DIR_EXTRA},
                1024: {False: TRAIN_DIR_1024, True: TRAIN_DIR_EXTRA_1024}
            }[self.crop_size][sample.is_external]

            for color in COLORS:
                try:
                    img = cv2.imread(f'{train_dir}/{image_id}_{color}.png', cv2.IMREAD_GRAYSCALE).astype("uint8")
                    # img = skimage.io.imread(f'{TRAIN_DIR}/{image_id}_{color}.png').copy()
                    images.append(img)
                except:
                    print(f'failed to open {train_dir}/{image_id}_{color}.png')
                    raise
            self.images[image_id] = images

        return [img.astype(np.float32) / 255.0 for img in self.images[image_id]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = self.load_images(sample)
        h, w = images[0].shape
        center_x = w / 2
        center_y = h / 2

        if self.is_training:
            geometry_aug_params = {
                0: GeometryAugLevel(prob=0.25, scale=0.1, scale_xy=0.1, angle=0, shear=0),
                10: GeometryAugLevel(prob=0.5, scale=0.5, scale_xy=0.25, angle=360, shear=15),
                20: GeometryAugLevel(prob=0.5, scale=0.7, scale_xy=0.25, angle=360, shear=15)
            }[self.geometry_aug_level]

            img_aug_params = {
                0: ImageAugLevel(brightness_common=0, brightness=0, gamma_common=0, gamma=0,
                                 mul_noise=0, mul_noise_p=0,
                                 blur_level=0, blur_p=0),
                10: ImageAugLevel(brightness_common=0.15, brightness=0.05, gamma_common=0.25, gamma=0.05,
                                  mul_noise=0, mul_noise_p=0,
                                  blur_level=0, blur_p=0),
                20: ImageAugLevel(brightness_common=0.15, brightness=0.1, gamma_common=0.25, gamma=0.1,
                                  mul_noise=0.25, mul_noise_p=0.25,
                                  blur_level=2.0, blur_p=0.2),
                30: ImageAugLevel(brightness_common=0.2, brightness=0.2, gamma_common=0.3, gamma=0.25,
                                  mul_noise=0.3, mul_noise_p=0.3,
                                  blur_level=2.0, blur_p=0.2),
            }[self.img_aug_level]

            scale = 2 ** np.random.normal(0, geometry_aug_params.scale)
            scale_x = 2 ** np.random.normal(0, geometry_aug_params.scale_xy)
            scale_y = 2 ** np.random.normal(0, geometry_aug_params.scale_xy)

            center_shift = 16
            cfg = TransformCfg(
                crop_size=self.crop_size,
                src_center_x=np.random.uniform(center_x - center_shift, center_x + center_shift),
                src_center_y=np.random.uniform(center_y - center_shift, center_y + center_shift),
                scale_x=scale * scale_x,
                scale_y=scale * scale_y,
                angle=np.random.uniform(0, geometry_aug_params.angle),
                shear=np.random.normal(0, geometry_aug_params.shear),
                hflip=np.random.choice([True, False]),
                vflip=np.random.choice([True, False])
            )

            # crops = [cfg.transform_image(img) for img in images]
            crops = []
            do_geom_transform = np.random.rand() < geometry_aug_params.prob
            for img in images:
                if do_geom_transform:
                    img = cfg.transform_image(img)

                brightness_common = 2 ** np.random.normal(0.0, img_aug_params.brightness_common)
                gamma_common = 2 ** np.random.normal(0.0, img_aug_params.gamma_common)

                if img_aug_params.gamma > 0:
                    brightness = 2 ** np.random.normal(0.0, img_aug_params.brightness)
                    gamma = 2 ** np.random.normal(0.0, img_aug_params.gamma)
                    img = brightness_common * brightness * (img ** (gamma_common * gamma))

                if img_aug_params.mul_noise_p > 0 and np.random.rand() < img_aug_params.mul_noise_p:
                    noise = 2 ** np.random.normal(0, img_aug_params.mul_noise, img.shape)
                    img = img * noise

                if img_aug_params.blur_p > 0 and np.random.rand() < img_aug_params.blur_p:
                    img = ndimage.gaussian_filter(img, np.random.rand() * img_aug_params.blur_level)

                crops.append(img)
        else:
            crops = images

        crops = np.array(crops).astype(np.float32)
        if self.transform:
            crops = self.transform(crops)

        return {
            'img': crops,
            'idx': idx,
            'image_id': sample.image_id,
            'labels': sample.labels
        }


class ClassificationDatasetTest(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.images = {}

        self.data = pd.read_csv('../input/sample_submission.csv')
        self.samples = list(self.data['Id'])

    def load_images(self, image_id):
        if image_id not in self.images:
            images = []
            for color in COLORS:
                img = skimage.io.imread(f'{TEST_DIR}/{image_id}_{color}.png').copy()
                images.append(img)
            self.images[image_id] = images

        return [img.astype(np.float32) / 255.0 for img in self.images[image_id]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = self.load_images(sample)
        crops = np.array(images).astype(np.float32)

        if self.transform:
            crops = self.transform(crops)

        return {
            'img': crops,
            'idx': idx,
            'image_id': sample
        }



def check_dataset():
    with utils.timeit_context('load ds'):
        ds = ClassificationDataset(fold=0, is_training=False)

    for sample in ds:
        plt.cla()
        plt.imshow(sample['img'])
        print(ds.samples[sample['idx']], sample['scale'])
        plt.show()


def check_dataset_aug():
    with utils.timeit_context('load ds'):
        ds = ClassificationDataset(fold=0, is_training=True, img_aug_level=20, geometry_aug_level=10)

    while True:
        sample = ds[1]
        utils.print_stats('img', sample['img'])
        plt.imshow(np.moveaxis(sample['img'], 0, 2)[:, :, :3])
        plt.show()


def check_color_dataset_aug():
    with utils.timeit_context('load ds'):
        ds = ClassificationDataset(fold=0, is_training=True)

    while True:
        sample = ds[1]
        plt.imshow(sample['img'])
        print(ds.samples[sample['idx']], sample['scale'])
        plt.show()


def check_dataset_performance():
    with utils.timeit_context('load ds'):
        ds = ClassificationDataset(fold=0, is_training=True)

    for sample in tqdm(ds):
        pass


if __name__ == '__main__':
    # check_dataset()
    # check_dataset_aug()
    # check_color_dataset_aug()
    check_dataset_performance()
