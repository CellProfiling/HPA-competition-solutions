import os, sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image
import cv2
from sklearn.utils import class_weight, shuffle
from ml_stratifiers import MultilabelStratifiedKFold
import warnings
warnings.filterwarnings("ignore")
from classification_models.resnet.models import ResNet18
import albumentations as A

MODEL_PATH = 'Christof/models/GAPNet/13_ext/'

# a) added batchnorm and cut out one Dense 256 layer
# b) a) + added 16 size layer to GAP
exp_suffix = '_4_2'
SIZE = 512

# Load dataset info
path_to_train = 'Christof/assets/train_rgb_512/'
data = pd.read_csv('Christof/assets/train.csv')

normal_aug = A.Compose([#A.Rotate((0,30),p=0.75),
                        A.RandomRotate90(p=1),
                        A.HorizontalFlip(p=0.5),
                        #A.RandomBrightness(0.05),
                        #A.RandomContrast(0.05),
                        A.IAAAffine(translate_percent=10,rotate=45,shear=10, scale=(0.9,1.1)),
                        #A.RandomAffine(degrees=45, translate=(0.1,0.1), shear=10, scale=(0.9,1.1))
                        A.Normalize(mean=(0.08069, 0.05258, 0.05487), std=(0.1300, 0.0879, 0.1386),
                                    max_pixel_value=255.)
                        ])

normal_aug_ext = A.Compose([#A.Rotate((0,30),p=0.75),
                        A.RandomRotate90(p=1),
                        A.HorizontalFlip(p=0.5),
                        #A.RandomBrightness(0.05),
                        #A.RandomContrast(0.05),
                        A.IAAAffine(translate_percent=10,rotate=45,shear=10, scale=(0.9,1.1)),
                        #A.RandomAffine(degrees=45, translate=(0.1,0.1), shear=10, scale=(0.9,1.1))
                        A.Normalize(mean=(0.1174382,  0.06798691, 0.06592218), std=(0.16392466 ,0.10036821, 0.16703453),
                                    max_pixel_value=255.)
                        ])

val_aug = A.Compose([A.HorizontalFlip(p=0.5),
                     A.Normalize(mean=(0.08069, 0.05258, 0.05487), std=(0.1300, 0.0879, 0.1386),
                                 max_pixel_value=255.)])
from torchvision import transforms

eps = 0.004
desired = {
    0: 0.36239782,
    1: 0.043841336,
    2: 0.075268817,
    3: 0.059322034,
    4: 0.075268817,
    5: 0.075268817,
    6: 0.043841336,
    7: 0.075268817,
    8: eps,
    9: eps,
    10: eps,
    11: 0.043841336,
    12: 0.043841336,
    13: 0.014198783,
    14: 0.043841336,
    15: eps,
    16: 0.028806584,
    17: 0.014198783,
    18: 0.028806584,
    19: 0.059322034,
    20: eps,
    21: 0.126126126,
    22: 0.028806584,
    23: 0.075268817,
    24: eps,
    25: 0.222493888,
    26: 0.028806584,
    27: eps
}

sampling_weights = [   2.6473,   35.0588 ,   8.2069  , 19.3439 ,  16.0145 ,  13.3245 ,  32.8644,
   10.607 ,  551.3   ,  501.1818 , 787.5714 ,  25.8523 ,  39.0301,   51.644,
   30.0846 ,1470.1333 ,  62.8262,  190.1034 ,  39.3084 ,  23.2126 , 170.9457
,    8.2592,   33.2609 ,   9.6889  , 92.2678  ,  4.19 ,    99.3333 ,3150.2857]

sample_weights_ext = [   2.6728,   41.1617 ,  10.3068 ,  42.4172  , 22.9729 ,  21.9808  , 26.8267
,   11.5358 , 474.8659 , 486.7375 , 492.8987  , 66.963 ,   50.2763  , 82.7609,
   45.0683, 1854.2381,  100.3582 , 319.1721  , 76.5762  , 33.424 ,  272.3007,
    7.3664 ,  39.4319  , 10.239 ,  734.6981 ,   2.548 ,  196.6616 , 638.3443]


train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    path = os.path.join(path_to_train, name)
    labs = np.array([int(label) for label in labels])
    bucket_ind = np.argmin([desired[l] for l in labs])
    bucket = labs[bucket_ind]
    weight = sampling_weights[bucket]
    train_dataset_info.append({
        'path': path,
        'labels': labs,
        'weight':weight})
train_dataset_info = np.array(train_dataset_info)


mskf = MultilabelStratifiedKFold(n_splits=5,shuffle=True,random_state=18)

y = np.zeros((len(train_dataset_info), 28))
for i in range(len(train_dataset_info)):
    y[i][train_dataset_info[i]['labels']] = 1
mskf.get_n_splits(train_dataset_info, y)
kf = mskf.split(train_dataset_info, y)
fold_infos = {}
for f in range(5):
    train_indexes, valid_indexes = next(kf)
    for item in train_dataset_info[valid_indexes]:
        fold_infos[item['path']] = f

folds_df = pd.DataFrame.from_dict(fold_infos,orient='index')
folds_df = folds_df.reset_index()
folds_df.columns = ['Id','fold']
folds_df['Id'] = folds_df['Id'].apply(lambda x : x.split('/')[-1])
folds_df.to_csv('stratified_5fold.csv',index=False)