import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
from multiprocessing import Pool


target_path = 'Christof/assets/ext_tomomi_rgby/'
input_path = 'Christof/assets/external_data6/'

data_ext1 = pd.read_csv('Christof/assets/train_ext1.csv')

target_size = 512

ids = [fn[:-5] for fn in data_ext1['Id']]


def save_rgb_512(id):
    image_red_ch = Image.open(input_path + id + '_red.jpg')
    image_green_ch = Image.open(input_path + id + '_green.jpg')
    image_blue_ch = Image.open(input_path + id + '_blue.jpg')
    image_yellow_ch = Image.open(input_path + id + '_yellow.jpg').convert('L')
    rgb_image = np.stack((
        np.array(image_red_ch)[:,:,0],
        np.array(image_green_ch)[:,:,1],
        np.array(image_blue_ch)[:,:,2],
        np.array(image_yellow_ch)), -1)

    rgb_image = cv2.resize(rgb_image, (target_size, target_size))
    cv2.imwrite(target_path + id + '.png', rgb_image)

p = Pool(8)
print(p.map(save_rgb_512, ids))
