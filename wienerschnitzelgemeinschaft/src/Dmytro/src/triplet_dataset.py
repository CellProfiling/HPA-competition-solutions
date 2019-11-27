import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict
from collections import namedtuple
from copy import copy

from config import *

TrainingSample = namedtuple('TrainingSample', 'image_id group')


class TripletDataset(Dataset):
    def __init__(self, is_train, transform=None, crop_size=256, embeddings_size=NB_EMBEDDINGS):
        self.transform = transform
        self.images = {}
        self.crop_size = crop_size
        self.is_train = is_train

        self.samples = []
        self.samples_idx_by_group = defaultdict(list)

        data = pd.read_csv('../input/folds_4_extra.csv')
        nb_train_samples = len(data) * 9 / 10

        for idx, row in data.iterrows():
            if is_train == (idx < nb_train_samples):
                image_id = row['Id']
                group = image_id.rsplit('_', 1)[0]
                sample = TrainingSample(image_id=image_id, group=group)
                self.samples.append(sample)
                self.samples_idx_by_group[group].append(len(self.samples) - 1)

        self.embeddings = np.zeros((len(self.samples), embeddings_size))

    def load_images(self, sample):
        image_id = sample.image_id
        if image_id not in self.images:
            images = []
            train_dir = TRAIN_DIR_EXTRA

            for color in ['red', 'green', 'blue']:
                try:
                    img = cv2.imread(f'{train_dir}/{image_id}_{color}.png', cv2.IMREAD_UNCHANGED)
                    img = cv2.resize(img,
                                     (self.crop_size, self.crop_size),
                                     interpolation=cv2.INTER_AREA).astype("uint8")
                    images.append(img)
                except:
                    print(f'failed to open {train_dir}/{image_id}_{color}.png')
                    raise
            self.images[image_id] = np.stack(images, axis=2)

        return self.images[image_id]  # .astype(np.float32) / 255.0

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        positive_samples = copy(self.samples_idx_by_group[sample.group])
        if len(positive_samples) > 1:
            positive_samples.remove(idx)

        positive_idx = np.random.choice(positive_samples)

        nb_negative_candidates = 1024
        negative_candidates = np.random.randint(len(self.samples), size=nb_negative_candidates)

        distances = np.linalg.norm(self.embeddings[negative_candidates]-self.embeddings[idx])
        negative_idx = negative_candidates[np.argmin(distances)]

        images = self.load_images(sample)
        images_pos = self.load_images(self.samples[positive_idx])
        images_neg = self.load_images(self.samples[negative_idx])

        def aug_image(img):
            if np.random.choice([True, False]):
                img = np.transpose(img, (1, 0, 2))

            if np.random.choice([True, False]):
                img = np.flip(img, 0)

            if np.random.choice([True, False]):
                img = np.flip(img, 1)

            return img.copy()

        if self.is_train:
            images = aug_image(images)
            images_pos = aug_image(images_pos)
            images_neg = aug_image(images_neg)

        if self.transform:
            images = self.transform(images)
            images_pos = self.transform(images_pos)
            images_neg = self.transform(images_neg)

        return {
            'img': images,
            'img_pos': images_pos,
            'img_neg': images_neg,
            'idx': idx,
            'image_id': sample.image_id,
        }


class TripletDatasetPredict(Dataset):
    def __init__(self, sample_ids, img_dir, transform=None, crop_size=256, embeddings_size=NB_EMBEDDINGS):
        self.transform = transform
        self.images = {}
        self.crop_size = crop_size

        self.samples = list(sample_ids)
        self.img_dir = img_dir

    def load_images(self, image_id):
        images = []

        for color in ['red', 'green', 'blue']:
            try:
                img = cv2.imread(f'{self.img_dir}/{image_id}_{color}.png', cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img,
                                 (self.crop_size, self.crop_size),
                                 interpolation=cv2.INTER_AREA).astype("uint8")
                images.append(img)
            except:
                print(f'failed to open {self.img_dir}/{image_id}_{color}.png')
                raise
        return np.stack(images, axis=2)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id = self.samples[idx]

        images = self.load_images(image_id)
        if self.transform:
            images = self.transform(images)

        return {
            'img': images,
            'idx': idx,
            'image_id': image_id,
        }


class TripletDatasetUpdate(Dataset):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        sample = self.train_dataset.samples[idx]

        images = self.train_dataset.load_images(sample)
        if self.train_dataset.transform:
            images = self.train_dataset.transform(images)

        return {
            'img': images,
            'idx': idx,
            'image_id': sample.image_id,
        }
