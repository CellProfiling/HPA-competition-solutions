import math
import numpy as np
import cv2
import torch
from config.config import *
from imgaug import augmenters as iaa

def train_multi_augment2(image):
    augment_func_list = [
        lambda image: (image), # default
        augment_flipud,                    # up-down
        augment_fliplr,                    # left-right
        augment_transpose,                 # transpose
        augment_flipud_lr,                 # up-down left-right
        augment_flipud_transpose,          # up-down transpose
        augment_fliplr_transpose,          # left-right transpose
        augment_flipud_lr_transpose,       # up-down left-right transpose
    ]
    c = np.random.choice(len(augment_func_list))
    image = augment_func_list[c](image)
    return image

def augment_default(image, mask=None):
    if mask is None:
        return image
    else:
        return image, mask

def augment_flipud(image, mask=None):
    image = np.flipud(image)
    if mask is None:
        return image
    else:
        mask = np.flipud(mask)
        return image, mask

def augment_fliplr(image, mask=None):
    image = np.fliplr(image)
    if mask is None:
        return image
    else:
        mask = np.fliplr(mask)
        return image, mask

def augment_transpose(image, mask=None):
    image = np.transpose(image, (1, 0, 2))
    if mask is None:
        return image
    else:
        if len(mask.shape) == 2:
            mask = np.transpose(mask, (1, 0))
        else:
            mask = np.transpose(mask, (1, 0, 2))
        return image, mask

def augment_flipud_lr(image, mask=None):
    image = np.flipud(image)
    image = np.fliplr(image)
    if mask is None:
        return image
    else:
        mask = np.flipud(mask)
        mask = np.fliplr(mask)
        return image, mask

def augment_flipud_transpose(image, mask=None):
    if mask is None:
        image = augment_flipud(image, mask=mask)
        image = augment_transpose(image, mask=mask)
        return image
    else:
        image, mask = augment_flipud(image, mask=mask)
        image, mask = augment_transpose(image, mask=mask)
        return image, mask

def augment_fliplr_transpose(image, mask=None):
    if mask is None:
        image = augment_fliplr(image, mask=mask)
        image = augment_transpose(image, mask=mask)
        return image
    else:
        image, mask = augment_fliplr(image, mask=mask)
        image, mask = augment_transpose(image, mask=mask)
        return image, mask

def augment_flipud_lr_transpose(image, mask=None):
    if mask is None:
        image = augment_flipud(image, mask=mask)
        image = augment_fliplr(image, mask=mask)
        image = augment_transpose(image, mask=mask)
        return image
    else:
        image, mask = augment_flipud(image, mask=mask)
        image, mask = augment_fliplr(image, mask=mask)
        image, mask = augment_transpose(image, mask=mask)
        return image, mask
