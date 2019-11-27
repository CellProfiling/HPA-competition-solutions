import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


input_path = 'assets/train/'
data = pd.read_csv('assets/train.csv')
target_size = 256
target_path = 'assets/train_256/'

ids = data['Id']
for id in tqdm(ids):
    image_red_ch = Image.open(input_path + id + '_red.png')
    image_green_ch = Image.open(input_path + id  + '_green.png')
    image_blue_ch = Image.open(input_path + id  + '_blue.png')
    image_yellow_ch = Image.open(input_path + id + '_yellow.png')
    rgba_image = np.stack((
        np.array(image_red_ch),
        np.array(image_green_ch),
        np.array(image_blue_ch),
        np.array(image_yellow_ch)), -1)

    rgba_image = cv2.resize(rgba_image, (target_size, target_size))
    cv2.imwrite(target_path + id + '.png',rgba_image)


# to open afterwards
#for id in tqdm(ids):
#    test2 = cv2.imread('tmp/' + id + '.png',cv2.IMREAD_UNCHANGED)

input_path = 'assets/test/'
data = pd.read_csv('assets/sample_submission.csv')
target_size = 256
target_path = 'assets/test_256/'

ids = data['Id']
for id in tqdm(ids):
    image_red_ch = Image.open(input_path + id + '_red.png')
    image_green_ch = Image.open(input_path + id  + '_green.png')
    image_blue_ch = Image.open(input_path + id  + '_blue.png')
    image_yellow_ch = Image.open(input_path + id + '_yellow.png')
    rgba_image = np.stack((
        np.array(image_red_ch),
        np.array(image_green_ch),
        np.array(image_blue_ch),
        np.array(image_yellow_ch)), -1)

    rgba_image = cv2.resize(rgba_image, (target_size, target_size))
    cv2.imwrite(target_path + id + '.png',rgba_image)


image = cv2.imread(target_path + id + '.png', cv2.IMREAD_UNCHANGED)