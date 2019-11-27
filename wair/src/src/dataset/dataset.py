from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
import torch.utils.data as data
import torch
import numpy as np
import cv2
from tqdm import tqdm
import random
import albumentations
import json

with open('SETTINGS.json', 'r') as f:
    path_dict = json.load(f)

train_transform = albumentations.Compose([

    albumentations.OneOf([
        albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
        albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.9),
        albumentations.RandomBrightness(limit=0.2, p=0.9),
        albumentations.RandomContrast(limit=0.2, p=0.9)
        ]),

    albumentations.Flip(p=0.75),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=180, interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_REFLECT_101, p=1),
    albumentations.OneOf([
        albumentations.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=1),
        albumentations.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=1)
    ], p=0.5),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
])

val_transform = albumentations.Compose([
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
])


class HPAIC_dataset_alb(data.Dataset):
    def __init__(self,
                 df = None,
                 name_list = None,
                 transform = None
                 ):
        self.df = df
        self.name_list = name_list
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]

        image = cv2.imread(path_dict['ALL_IMAGE_TRAIN_DATA_CLEAN_DIR'] + '{name}.png'.format(name=name), -1)
        image = image[:,:,0:3]
        label = torch.FloatTensor(self.df[self.df['Id']==name].loc[:, 'Nucleoplasm':'Rods & rings'].values)

        image = self.transform(image=image)['image'].transpose(2, 0, 1)

        return image, label


def generate_dataset_loader(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers):

    train_dataset = HPAIC_dataset_alb(df_all, c_train, train_transform)
    val_dataset = HPAIC_dataset_alb(df_all, c_val, val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,        
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,        
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False)

    return train_loader, val_loader


def read_list_from_txt(file_name):

    f_dataset = open(file_name, 'r')
    c_dataset = f_dataset.readlines()
    f_dataset.close()
    c_dataset = [s.replace('\n', '') for s in c_dataset]

    return c_dataset
    
# There are three versions of oversampling.(v2, v3, v4)
def balance_class_process_all_more_train_add_3_v2(c_train, c_val):

    file_path = 'class_txt_all_more_train/'

    # load all names of samples for rare classes 
    class_15_list = read_list_from_txt(file_path + 'class_15_37.txt')
    class_10_list = read_list_from_txt(file_path + 'class_10_105.txt')
    class_9_list = read_list_from_txt(file_path + 'class_9_116.txt')
    class_8_list = read_list_from_txt(file_path + 'class_8_159.txt')
    class_27_list = read_list_from_txt(file_path + 'class_27_53.txt')
    class_20_list = read_list_from_txt(file_path + 'class_20_319.txt')
    class_17_list = read_list_from_txt(file_path + 'class_17_267.txt')
    class_24_list = read_list_from_txt(file_path + 'class_24_472.txt')
    class_26_list = read_list_from_txt(file_path + 'class_26_593.txt')

    c_train_new = []
    for name in c_train:

        if name in class_27_list:
            for i in range(10):
                c_train_new.append(name)
        elif name in class_15_list:
            for i in range(10):
                c_train_new.append(name)
        elif name in class_10_list:
            for i in range(5):
                c_train_new.append(name)
        elif name in class_9_list:
            for i in range(5):
                c_train_new.append(name)
        elif name in class_8_list:
            for i in range(4):
                c_train_new.append(name)
        elif name in class_20_list:
            for i in range(3):
                c_train_new.append(name)
        elif name in class_17_list:
            for i in range(3):
                c_train_new.append(name)
        elif name in class_24_list:
            for i in range(2):
                c_train_new.append(name)
        elif name in class_26_list:
            for i in range(2):
                c_train_new.append(name)
        else:
            c_train_new.append(name)

    return c_train_new, c_val

def balance_class_process_all_more_train_add_3_v3(c_train, c_val):

    file_path = 'class_txt_all_more_train/'

    # load all names of samples for rare classes 
    class_15_list = read_list_from_txt(file_path + 'class_15_37.txt')
    class_10_list = read_list_from_txt(file_path + 'class_10_105.txt')
    class_9_list = read_list_from_txt(file_path + 'class_9_116.txt')
    class_8_list = read_list_from_txt(file_path + 'class_8_159.txt')
    class_27_list = read_list_from_txt(file_path + 'class_27_53.txt')
    class_20_list = read_list_from_txt(file_path + 'class_20_319.txt')
    class_17_list = read_list_from_txt(file_path + 'class_17_267.txt')
    class_24_list = read_list_from_txt(file_path + 'class_24_472.txt')
    class_26_list = read_list_from_txt(file_path + 'class_26_593.txt')
    class_16_list = read_list_from_txt(file_path + 'class_16_877.txt')
    class_18_list = read_list_from_txt(file_path + 'class_18_1436.txt')
    class_22_list = read_list_from_txt(file_path + 'class_22_1789.txt')
    class_26_list = read_list_from_txt(file_path + 'class_26_593.txt')

    # load names of samples which belong to class 0 in external data
    class_0_only_list = read_list_from_txt(file_path + 'class_0_only_extral.txt')
    # load names of samples which belong to class 25 in external data
    class_25_only_list = read_list_from_txt(file_path + 'class_25_only_extral.txt')
    # load names of samples which belong to class 0 and 25 in external data
    class_0_25_only_list = read_list_from_txt(file_path + 'class_0_25_only_extral.txt')

    c_train_new = []
    for name in c_train:

        if name in class_27_list:
            for i in range(10):
                c_train_new.append(name)
        elif name in class_15_list:
            for i in range(10):
                c_train_new.append(name)
        elif name in class_10_list:
            for i in range(4):
                c_train_new.append(name)
        elif name in class_9_list:
            for i in range(4):
                c_train_new.append(name)
        elif name in class_8_list:
            for i in range(4):
                c_train_new.append(name)
        elif name in class_20_list:
            for i in range(3):
                c_train_new.append(name)
        elif name in class_17_list:
            for i in range(3):
                c_train_new.append(name)
        elif name in class_16_list:
            for i in range(2):
                c_train_new.append(name)
        elif name in class_18_list:
            for i in range(2):
                c_train_new.append(name)
        elif name in class_22_list:
            for i in range(2):
                c_train_new.append(name)
        elif name in class_26_list:
            for i in range(2):
                c_train_new.append(name)
        # elif name in class_0_only_list + class_25_only_list + class_0_25_only_list:
        #     pass
        else:
            c_train_new.append(name)

    return c_train_new, c_val

def balance_class_process_all_more_train_add_3_v4(c_train, c_val):

    file_path = 'class_txt_all_more_train/'

    # load all names of samples for rare classes 
    class_15_list = read_list_from_txt(file_path + 'class_15_37.txt')
    class_10_list = read_list_from_txt(file_path + 'class_10_105.txt')
    class_9_list = read_list_from_txt(file_path + 'class_9_116.txt')
    class_8_list = read_list_from_txt(file_path + 'class_8_159.txt')
    class_27_list = read_list_from_txt(file_path + 'class_27_53.txt')
    class_20_list = read_list_from_txt(file_path + 'class_20_319.txt')
    class_17_list = read_list_from_txt(file_path + 'class_17_267.txt')
    class_24_list = read_list_from_txt(file_path + 'class_24_472.txt')
    class_26_list = read_list_from_txt(file_path + 'class_26_593.txt')
    class_16_list = read_list_from_txt(file_path + 'class_16_877.txt')
    class_18_list = read_list_from_txt(file_path + 'class_18_1436.txt')
    class_22_list = read_list_from_txt(file_path + 'class_22_1789.txt')
    class_26_list = read_list_from_txt(file_path + 'class_26_593.txt')

    # load names of samples which belong to class 0 in external data
    class_0_only_list = read_list_from_txt(file_path + 'class_0_only_extral.txt')
    # load names of samples which belong to class 25 in external data
    class_25_only_list = read_list_from_txt(file_path + 'class_25_only_extral.txt')
    # load names of samples which belong to class 0 and 25 in external data
    class_0_25_only_list = read_list_from_txt(file_path + 'class_0_25_only_extral.txt')

    c_train_new = []
    for name in c_train:

        if name in class_27_list:
            for i in range(10):
                c_train_new.append(name)
        elif name in class_15_list:
            for i in range(10):
                c_train_new.append(name)
        elif name in class_10_list:
            for i in range(4):
                c_train_new.append(name)
        elif name in class_9_list:
            for i in range(4):
                c_train_new.append(name)
        elif name in class_8_list:
            for i in range(4):
                c_train_new.append(name)
        elif name in class_20_list:
            for i in range(3):
                c_train_new.append(name)
        elif name in class_17_list:
            for i in range(3):
                c_train_new.append(name)
        elif name in class_16_list:
            for i in range(2):
                c_train_new.append(name)
        elif name in class_18_list:
            for i in range(2):
                c_train_new.append(name)
        elif name in class_22_list:
            for i in range(2):
                c_train_new.append(name)
        elif name in class_26_list:
            for i in range(2):
                c_train_new.append(name)
        elif name in class_0_only_list + class_25_only_list + class_0_25_only_list:
            pass
        else:
            c_train_new.append(name)

    return c_train_new, c_val

