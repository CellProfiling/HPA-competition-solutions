import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


input_path = 'Christof/assets/ext_tomomi/'
data = pd.read_csv('Christof/assets/train.csv')
target_size = 512
target_path = 'Christof/assets/train_rgb_512_ext1/'

ids = os.listdir(input_path)
for id in tqdm(ids):
    image = Image.open(input_path + id)
    image = np.array(image)
    #image = image[:,:,[2,1,0]]
    #rgba_image = cv2.resize(np.array(image), (target_size, target_size))
    cv2.imwrite(target_path + id,image)
