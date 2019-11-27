import pandas as pd
import cv2
from skimage.io import imread, imsave
import os
from os.path import join as pjoin
from lib.utils import multiprocessing, load_img
import numpy as np
from lib.utils import str_to_labels

CLASS_ID = 17

df = pd.read_csv('data/train_with_hpa.csv', index_col=0)

folder_to_original = f'tmp/selected_imgs_{CLASS_ID}/original'
folder_to_green = f'tmp/selected_imgs_{CLASS_ID}/green'

os.makedirs(folder_to_original, exist_ok=True)
os.makedirs(folder_to_green, exist_ok=True)

df_filtered = df.loc[[CLASS_ID in str_to_labels(x) for x in df['Target']]]


def processing_one_img(task):
    id_, row = task

    img = load_img(
        id_,
        resize=None,
        group=row['group'],
        channels=['red', 'green', 'blue'],
    )

    img_fn = f"{id_}.jpg"

    imsave(pjoin(folder_to_original, img_fn), img)
    img_green = img[:, :, 1]
    imsave(pjoin(folder_to_green, img_fn), img_green)

    return {
        'id_': id_,
        'target': row['Target'],
    }


pd.DataFrame\
  .from_records(multiprocessing(processing_one_img, df_filtered.iterrows()))\
  .sort_values('id_')\
  .to_csv(f'tmp/selected_imgs_{CLASS_ID}.csv', index=False)
