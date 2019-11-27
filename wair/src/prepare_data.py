import torch.utils.data as data
import torch
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import json
import os

# convert external data to RGBY images
def image_read_4channel_external(path, name):
    image_red = cv2.imread(path + '/%s_red.jpg' % name, 0)
    image_yellow = cv2.imread(path + '/%s_yellow.jpg' % name, 0)
    image_blue = cv2.imread(path + '/%s_blue.jpg' % name, 0)
    image_green = cv2.imread(path + '/%s_green.jpg' % name, 0)
    if image_yellow is None:
        image_yellow = np.zeros_like(image_red)

    image_red = cv2.normalize(image_red,None,0,255,cv2.NORM_MINMAX)
    image_red = cv2.resize(image_red, (512, 512))
    image_red = image_red.astype(np.uint8)

    image_yellow = cv2.normalize(image_yellow,None,0,255,cv2.NORM_MINMAX)
    image_yellow = cv2.resize(image_yellow, (512, 512))
    image_yellow = image_yellow.astype(np.uint8)
    
    image_blue = cv2.normalize(image_blue,None,0,255,cv2.NORM_MINMAX)
    image_blue = cv2.resize(image_blue, (512, 512))
    image_blue = image_blue.astype(np.uint8)
    
    image_green = cv2.normalize(image_green,None,0,255,cv2.NORM_MINMAX)
    image_green = cv2.resize(image_green, (512, 512))
    image_green = image_green.astype(np.uint8)
    image = np.stack([image_red,image_green,image_blue,image_yellow]).transpose(1,2,0)

    return image

# convert kaggle data to RGBY images
def image_read_4channel_kaggle(path, name):
    image_red = cv2.imread(path + '/%s_red.png' % name, 0)
    image_yellow = cv2.imread(path + '/%s_yellow.png' % name, 0)
    image_blue = cv2.imread(path + '/%s_blue.png' % name, 0)
    image_green = cv2.imread(path + '/%s_green.png' % name, 0)

    image_red = image_red.astype(np.uint8)
    image_yellow = image_yellow.astype(np.uint8)
    image_blue = image_blue.astype(np.uint8)
    image_green = image_green.astype(np.uint8)
    image = np.stack([image_red,image_green,image_blue,image_yellow]).transpose(1,2,0)

    return image

class HPAIC_process_external(data.Dataset):
    def __init__(self,
                 name_list = None,
                 transform = None,
                 read_path = None,
                 write_path = None
                 ):
        self.name_list = name_list
        self.transform = transform
        self.read_path = read_path
        self.write_path = write_path

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]

        image = image_read_4channel_external(self.read_path, name)
        cv2.imwrite(self.write_path + name + '.png', image)
        return name

class HPAIC_process_kaggle(data.Dataset):
    def __init__(self,
                 name_list = None,
                 transform = None,
                 read_path = None,
                 write_path = None
                 ):
        self.name_list = name_list
        self.transform = transform
        self.read_path = read_path
        self.write_path = write_path

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]

        image = image_read_4channel_kaggle(self.read_path, name)
        cv2.imwrite(self.write_path + name + '.png', image)
        return name


if __name__ == '__main__':

    with open('SETTINGS.json', 'r') as f:
        path_dict = json.load(f)

    RAW_DATA_DIR = path_dict['RAW_DATA_DIR']
    EXTERNAL_DATA_DIR = path_dict['EXTERNAL_DATA_DIR']
    ALL_IMAGE_TRAIN_DATA_CLEAN_DIR = path_dict['ALL_IMAGE_TRAIN_DATA_CLEAN_DIR']
    IMAGE_TEST_DATA_CLEAN_DIR = path_dict['IMAGE_TEST_DATA_CLEAN_DIR']

    if not os.path.exists(ALL_IMAGE_TRAIN_DATA_CLEAN_DIR):
        os.makedirs(ALL_IMAGE_TRAIN_DATA_CLEAN_DIR)
    if not os.path.exists(IMAGE_TEST_DATA_CLEAN_DIR):
        os.makedirs(IMAGE_TEST_DATA_CLEAN_DIR)
        
    kaggle_train_image_list = glob(RAW_DATA_DIR + 'train/*')
    kaggle_train_image_list = list(set([s.split('/')[-1].replace('.png', '').replace('_red', '').replace('_yellow', '').replace('_blue', '').replace('_green', '') for s in kaggle_train_image_list]))
    
    kaggle_test_image_list = glob(RAW_DATA_DIR + 'test/*')
    kaggle_test_image_list = list(set([s.split('/')[-1].replace('.png', '').replace('_red', '').replace('_yellow', '').replace('_blue', '').replace('_green', '') for s in kaggle_test_image_list]))

    external_image_list = glob(EXTERNAL_DATA_DIR + 'jpgs/*')
    external_image_list = list(set([s.split('/')[-1].replace('.jpg', '').replace('_red', '').replace('_yellow', '').replace('_blue', '').replace('_green', '') for s in external_image_list]))

    print('kaggle train images are processing')
    train_dataset = HPAIC_process_kaggle(name_list=kaggle_train_image_list, 
                                         read_path=RAW_DATA_DIR + 'train',
                                         write_path=ALL_IMAGE_TRAIN_DATA_CLEAN_DIR)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,        
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=False)
    num = 0
    for batchID, (name) in enumerate (train_loader):
        num = num + 16

    print('Done')
    print('kaggle test images are processing')
    train_dataset = HPAIC_process_kaggle(name_list=kaggle_test_image_list, 
                                         read_path=RAW_DATA_DIR + 'test',
                                         write_path=IMAGE_TEST_DATA_CLEAN_DIR)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,        
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=False)
    num = 0
    for batchID, (name) in enumerate (train_loader):
        num = num + 16

    print('Done')
    print('external images are processing')
    train_dataset = HPAIC_process_external(name_list=external_image_list, 
                                         read_path=EXTERNAL_DATA_DIR + 'jpgs',
                                         write_path=ALL_IMAGE_TRAIN_DATA_CLEAN_DIR)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,        
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=False)
    num = 0
    for batchID, (name) in enumerate (train_loader):
        num = num + 16
    print('Done')