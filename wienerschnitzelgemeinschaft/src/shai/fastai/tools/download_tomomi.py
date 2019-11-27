import requests
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from PIL import Image
import requests
from io import BytesIO

newpath = "/media/galib/Documents/protein-atlas/jpg/"
if not os.path.exists(newpath):
    os.makedirs(newpath)

print(newpath)

colors = ['red','green','blue','yellow']
#DIR = "/media/galib/Documents/protein-atlas/jpg/"
v18_url = 'http://v18.proteinatlas.org/images/'

imgList = pd.read_csv("../input/HPAv18/HPAv18RBGY_wodpl.csv")

print(len(imgList))

urls = []
names = []
for i in tqdm(imgList['Id']): # [:5] means downloard only first 5 samples, if it works, please remove it
    img = i.split('_')
    for color in colors:
        img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
        img_name = i + "_" + color + ".jpg"
        img_url = v18_url + img_path
        #r = requests.get(img_url, allow_redirects=True)
        #open(DIR + img_name, 'wb').write(r.content)

        urls.append(img_url)
        names.append(img_name)

def load_img2(url,name):
    fname = newpath+name
    if not os.path.exists(fname):
        response = requests.get(url)
    #     print(response)
        img = Image.open(BytesIO(response.content))
    #     print(img.__dict__)
        img.save(fname)
    #     print(fname)

# download the images in parallel
num_cores = 8
Parallel(n_jobs=num_cores, prefer="threads")(delayed(load_img2)(u,n) \
                                             for u,n in tqdm(zip(urls,names), total=len(names)))

print('done')