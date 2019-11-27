# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 21:34:52 2018

@author: Xuan-Laptop
"""

import os

import cv2
import numpy as np
import torch
from torch.utils import data
from Augmentation import*
from sklearn.preprocessing import MultiLabelBinarizer

def img_to_torch(image):
    return torch.from_numpy(np.transpose(image, (2, 0, 1)).astype('float32'))

def augment(image):
    # TODO implement augmentation
    pass

def encoding(x):
    labels = np.array(list(map(int, x.split(' '))))
    #y = np.eye(28, dtype=np.float)[labels].sum(axis=0)
    return labels

class HPADataset(data.Dataset):
    def __init__(self, root_path, images_df, size, mode='train', transforms=False, color='rgb',
                 ext_path=None, range=1, test_tta=None):
        self.mode = mode
        self.root_path = root_path
        self.ext_path = ext_path
        self.images_df = images_df
        self.transforms = transforms
        self.size = size
        self.color = color
        self.has_yellow = color.find('y')
        self.red = color.find('r')
        self.green = color.find('g')
        self.blue = color.find('b')
        self.range = range
        self.test_tta = test_tta

        if mode!='test':
            mlb = MultiLabelBinarizer()
            self.images_df['labels'] = self.images_df['Target'].apply(encoding)
            self.labels = mlb.fit_transform(self.images_df['labels'].tolist())
    
    def __len__(self):
        # modified by wyh
        return len(self.images_df)

    def set_tta(self, new_tta):
        if new_tta=='null':
            new_tta = None
        self.test_tta = new_tta
    
    def __getitem__(self, index):
        # modified by wyh
        if index not in range(0, len(self.images_df)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        # df_index = self.file_list[index]
        file_id = self.images_df.Id.iloc[index]
        ext = self.images_df.ext.iloc[index]
        if ext==0:
            image_path = os.path.join(self.root_path, file_id)
        else:
            image_path = os.path.join(self.ext_path, file_id)

        # only take on channel
        image = cv2.imread(str(image_path) + '.png', cv2.IMREAD_UNCHANGED)
        if self.size!=512:
            image = cv2.resize(image, (self.size, self.size))
        if image is None:
            print(file_id)

        if self.range==1:
            image = (image/255.0).astype(np.float32)

        if self.transforms:
            if np.random.rand()<0.75:
                image = do_random_clockwise_rotate(image)
            if np.random.rand()<0.5:
                image = do_horizontal_flip(image)
            if np.random.rand()<0.5:
                image= do_vertical_flip(image)

            if np.random.rand()<0.4:
                image = do_raodom_brightness_shift(image, alpha=0.1, range=self.range)
            if np.random.rand()<0.4:
                image = do_random_brightness_multiply(image, alpha=0.1, range=self.range)
            if np.random.rand()<0.4:
                image= randomGamma(image, gamma_limit=0.1, range=self.range)

            if np.random.rand()<0.5:
                image = do_horizontal_shear2(image, dx=0.05)

        if self.mode == 'test' or self.mode=='valid':
            if self.test_tta=='hflip':
                image = do_horizontal_flip(image)
            if self.test_tta=='vflip':
                image = do_vertical_flip(image)
            if self.test_tta=='bflip':
                image = do_vertical_flip(image)
                image = do_horizontal_flip(image)
            if self.test_tta =='rotate':
                image = do_random_clockwise_rotate(image)
            if self.test_tta =='gamma':
                image = randomGamma(image, gamma_limit=0.1, range=self.range)
            if self.mode=='test':
                return img_to_torch(image)
            else:
                return (img_to_torch(image), self.labels[index])

        elif self.mode == 'train':
            return (img_to_torch(image), self.labels[index])


