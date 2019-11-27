import cv2
import numpy as np
import pandas as pd
from tensorflow.python.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorpack.utils import logger
import config
import os
from tensorflow import keras
import ast
import math
from albumentations import (
    RandomCrop, HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma
)
from imgaug import augmenters as iaa
from PIL import Image
import pickle

def calc_macro_f1(predict, truth, threshold=0.5):
    EPS = 1e-12
    truth = truth > 0.5
    pp = []
    for p in predict:
        if (np.all(p <= threshold)):
            pp.append(p>=np.max(p))
        else:
            pp.append(p > threshold)
    predict = np.array(pp)
    tp = np.sum(truth & predict, axis=0)
    num_pos = np.sum(predict, axis=0)
    num_true = np.sum(truth, axis=0)
    precision = tp / (num_pos + EPS)
    recall = tp / (num_true + EPS)
    #fp = np.sum(~truth & predict, axis=0)
    #fn = np.sum(truth & ~predict, axis=0)

    #precision = tp / (tp + fp + EPS)
    #recall = tp / (tp + fn + EPS)

    f1_scores = (2 * precision * recall) / (precision + recall + EPS)

    return np.mean(f1_scores)

def strong_aug(p=1.0):
    return Compose([
        #RandomCrop(width=1024, height=1024, p=1.0)
        #RandomBrightness(p=0.5),
        #RandomContrast(p=0.5),
        RandomRotate90(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Transpose(p=0.5),
    ], p=p)

def preprocess(d, is_training=True):
    img_id, label = d
    if config.BIG:
        image4c = open_rgby_2048(img_id)
    else:
        image4c = open_rgby(config.TRAIN_DATASET, img_id)
    if is_training:
        augmented = strong_aug()(image=image4c)
        image4c = augmented['image']
    #if isinstance(label, list):
    #    label = np.eye(config.NUM_CLASS, dtype=np.float)[np.array(label)].sum(axis=0)
    return [image4c, label]

def open_rgby(path, id): #a function that reads RGBY image
    colors = ['red','green','blue','yellow'] if not config.RGB else ['red','green','blue']
    flags = cv2.IMREAD_GRAYSCALE
    #img = [cv2.imread(os.path.join(path, id+'_'+color+'.png'), flags)
    #       for color in colors]
    img = [cv2.imread(id+'_'+color+'.png', flags)
           for color in colors]
    image4c = np.stack(img, axis=-1)
    image4c = image4c.astype(np.uint8)
    return image4c

def open_pickle(_id):
    with open(_id + ".pkl", 'rb') as f:
        image = pickle.load(f)
        if config.RGB:
            return image[:,:,:3]
    return image

def open_rgby_2048(id):
    return open_pickle(id)