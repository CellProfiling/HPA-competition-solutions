import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

tile = 'br'

input_path = 'Christof/assets/train_rgb_1024/'
data = pd.read_csv('Christof/assets/train.csv')
target_size = 512
target_path = f'Christof/assets/train_rgb_1024_9crop/{tile}/'

ids = data['Id']

channel_avg = np.zeros(3)
channel_std = np.zeros(3)
for id in tqdm(ids):
    image = cv2.imread(input_path + id + '.png', cv2.IMREAD_UNCHANGED)
    cv2.imwrite(target_path + id + '.png',image[512:, 512:,:])
    img = np.reshape(image[512:, 512:,:],(-1,3))
    channel_avg += np.mean(img,axis=0)
    channel_std += np.std(img,axis=0)

channel_avg/=len(ids)
channel_std/=len(ids)

print(channel_avg/255)
print(channel_std/255)


input_path = 'Christof/assets/test_rgb_1024/'
data = pd.read_csv('Christof/assets/sample_submission.csv')
target_path = f'Christof/assets/test_rgb_1024_9crop/{tile}/'

ids = data['Id']

for id in tqdm(ids):
    image = cv2.imread(input_path + id +'.png', cv2.IMREAD_UNCHANGED)
    cv2.imwrite(target_path + id + '.png', image[512:, 512:, :])


input_path = 'Christof/assets/ext_tomomi_rgb_1024/'
data_ext1 = pd.read_csv('Christof/assets/train_ext1.csv')

ids = [fn[:-5] for fn in data_ext1['Id']]

target_path = f'Christof/assets/ext_tomomi_rgb_1024_9crop/{tile}/'

channel_avg = np.zeros(3)
channel_std = np.zeros(3)
for id in tqdm(ids):
    image = cv2.imread(input_path + id +'.png', cv2.IMREAD_UNCHANGED)
    cv2.imwrite(target_path + id + '.png',image[512:, 512:,:])
    img = np.reshape(image[512:, 512:,:],(-1,3))
    channel_avg += np.mean(img,axis=0)
    channel_std += np.std(img,axis=0)

channel_avg/=len(ids)
channel_std/=len(ids)

print(channel_avg/255)
print(channel_std/255)

