import cv2
import pandas as pd
from tqdm import tqdm
import os
train = pd.read_csv('Christof/assets/train_ext1.csv')
#test = pd.read_csv('Christof/assets/sample_submission.csv')

path_to_train = 'Christof/assets/ext_tomomi_rgb_256_ul/'
#path_to_test = 'Christof/assets/test_rgby_512/'

fns = os.listdir(path_to_train)
import numpy as np


data_ext1 = pd.read_csv('Christof/assets/train_ext1.csv')

ids = [fn[:-5] for fn in data_ext1['Id']]


channel_avg = np.zeros(3)
channel_std = np.zeros(3)
#images = np.zeros((len(fns),512,512,3))
for i, fn in tqdm(enumerate(ids)):
    image = cv2.imread(path_to_train + fn + '.png', cv2.IMREAD_UNCHANGED)
    channel_avg += np.mean(np.reshape(image,(-1,3)),axis=0)
    channel_std += np.std(np.reshape(image,(-1,3)),axis=0)


channel_avg/=len(fns)
channel_std/=len(fns)

print(channel_avg/255)
print(channel_std/255)