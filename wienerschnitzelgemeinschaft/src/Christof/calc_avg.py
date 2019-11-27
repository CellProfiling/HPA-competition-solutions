import cv2
import pandas as pd
from tqdm import tqdm

train = pd.read_csv('Christof/assets/train.csv')
test = pd.read_csv('Christof/assets/sample_submission.csv')

path_to_train = 'Christof/assets/train_rgb_1024_9crop/bl/'
path_to_test = 'Christof/assets/test_rgby_512/'

fns = [path_to_train + f + '.png' for f in train['Id']]

import numpy as np
channel_avg = np.zeros(3)
channel_std = np.zeros(3)
#images = np.zeros((len(fns),512,512,3))
for i, fn in tqdm(enumerate(fns)):
    image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    img = np.reshape(image,(-1,3))
    channel_avg += np.mean(img,axis=0)
    channel_std += np.std(img,axis=0)

channel_avg/=len(fns)
channel_std/=len(fns)

print(channel_avg/255)
print(channel_std/255)