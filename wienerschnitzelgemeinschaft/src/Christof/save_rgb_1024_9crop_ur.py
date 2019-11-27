import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


input_path = 'Christof/assets/train_rgb_1024/'
data = pd.read_csv('Christof/assets/train.csv')
target_size = 256
target_path = 'Christof/assets/train_rgb_1024_9crop/ur/'

ids = data['Id']


for id in tqdm(ids):
    image = cv2.imread(input_path + id + '.png', cv2.IMREAD_UNCHANGED)
    cv2.imwrite(target_path + id + '.png',image[:512, 512:,:])


input_path = 'Christof/assets/test_rgb_1024/'
data = pd.read_csv('Christof/assets/sample_submission.csv')
target_path = 'Christof/assets/test_rgb_1024_9crop/ur/'

ids = data['Id']

for id in tqdm(ids):
    image = cv2.imread(input_path + id +'.png', cv2.IMREAD_UNCHANGED)
    cv2.imwrite(target_path + id + '.png',image[:512, 512:,:])


input_path = 'Christof/assets/ext_tomomi_rgb_1024/'
data_ext1 = pd.read_csv('Christof/assets/train_ext1.csv')

ids = [fn[:-5] for fn in data_ext1['Id']]

target_path = 'Christof/assets/ext_tomomi_rgb_1024_9crop/ur/'

for id in tqdm(ids):
    image = cv2.imread(input_path + id +'.png', cv2.IMREAD_UNCHANGED)
    cv2.imwrite(target_path + id + '.png',image[:512, 512:,:])


