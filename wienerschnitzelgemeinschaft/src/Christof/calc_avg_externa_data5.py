import cv2
import pandas as pd
from tqdm import tqdm

train = pd.read_csv('Christof/assets/train_ext1.csv')
#test = pd.read_csv('Christof/assets/sample_submission.csv')

path_to_train = 'Christof/assets/external_data5/'
#path_to_test = 'Christof/assets/test_rgby_512/'

fns = [path_to_train + f[:-5] + '.png' for f in train['Id']]

import numpy as np


channel_avg = np.zeros(4)
channel_std = np.zeros(4)
#images = np.zeros((len(fns),512,512,3))
for i, fn in tqdm(enumerate(fns)):
    image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    channel_avg += np.mean(np.reshape(image,(-1,4)),axis=0)
    channel_std += np.std(np.reshape(image,(-1,4)),axis=0)


channel_avg/=len(fns)
channel_std/=len(fns)

print(channel_avg/255)
print(channel_std/255)