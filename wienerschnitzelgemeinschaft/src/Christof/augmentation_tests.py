import numpy as np
import cv2
from matplotlib import pyplot as plt
import albumentations as A
import pandas as pd

def augment_and_show(aug, image):
    image = aug(image=image)['image']
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

data = pd.read_csv('Christof/assets/train.csv').sample(frac=0.5)


fn = '4a32cfea-bbc3-11e8-b2bc-ac1f6b6435d0'
image = cv2.imread('Christof/assets/train_rgb_256/' + fn + '.png', cv2.IMREAD_UNCHANGED)

image2 = A.RandomCrop(image,200,200).targets['image']
plt.figure(figsize=(10, 10))
plt.imshow(image2)

image2 = A.RandomCrop(200,200)(image=image)['image']

image3 = A.ShiftScaleRotate(p=1)(image=image)['image']
plt.imshow(image3)