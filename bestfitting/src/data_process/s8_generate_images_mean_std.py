import sys
sys.path.insert(0, '..')
import cv2
import pandas as pd
import mlcrate as mlc

from config.config import *
from utils.common_util import *
from PIL import Image
from tqdm import tqdm

def get_img_mean_std(img_dir, color, img_mean, img_std):
    img_list = os.listdir(img_dir)
    img_list = [i for i in img_list if i.count(color) > 0]
    print(img_dir, len(img_list))
    for img_id in tqdm(img_list):
        image_path = opj(img_dir, img_id)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255

        m = img.mean()
        s = img.std()
        img_mean.append(m)
        img_std.append(s)
    return img_mean, img_std

if __name__ == "__main__":
    print('%s: calling main function ... ' % os.path.basename(__file__))

    for color in ['red', 'green', 'blue', 'yellow']:
        img_mean = []
        img_std = []
        img_dir = opj(DATA_DIR, 'test/images')
        img_mean, img_std = get_img_mean_std(img_dir, color, img_mean, img_std)
        img_dir = opj(DATA_DIR, 'train/images')
        img_mean, img_std = get_img_mean_std(img_dir, color, img_mean, img_std)
        print(color, np.around(np.mean(img_mean), decimals=6), np.around(np.mean(img_std), decimals=6))

    print('\nsuccess!')
