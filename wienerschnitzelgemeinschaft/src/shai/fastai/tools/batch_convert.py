import cv2
import glob, os, errno
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed

img_dir = "/media/galib/Documents/protein-atlas/jpg/"
#img_dir = "/home/galib/kaggle/ProteinAtlas/input/HPAv18/jpg/"
savedir = '/home/galib/kaggle/ProteinAtlas/input/HPAv18/png'
if not os.path.exists(savedir):
    os.makedirs(savedir)

size = 512,512
colors = ['red','green','blue','yellow']
imgList = pd.read_csv("../input/HPAv18/HPAv18RBGY_wodpl.csv")
print(len(imgList))


#for i in tqdm(imgList['Id']):

def save_image(i):
    for color in colors:
        img_path = i + "_" + color + ".jpg"
        img_name = i + "_" + color + ".png"
        img_full_path = img_dir + img_path

        fname = os.path.join(savedir,img_name)
        if not os.path.exists(fname):
            x = Image.open(img_full_path, 'r')
            x = x.convert('L')  # makes it greyscale
            y = np.asarray(x.getdata(), dtype=np.float64).reshape((x.size[1], x.size[0]))
            y = np.asarray(y, dtype=np.uint8)  # if values still in range 0-255!
            w = Image.fromarray(y, mode='L')
            w.thumbnail(size, Image.ANTIALIAS)
            w.save(os.path.join(savedir,img_name))

num_cores = 6
Parallel(n_jobs=num_cores, prefer="threads")(delayed(save_image)(i) \
                                             for i in tqdm(imgList['Id']))

print('done')