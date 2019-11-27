"""
import requests
import pandas as pd
colors = ['red','green','blue','yellow']
DIR = "./HPAv18/jpg/"
v18_url = 'http://v18.proteinatlas.org/images/'
imgList = pd.read_csv("./HPAv18RBGY_wodpl.csv")
for idx, i in enumerate(imgList['Id'][41093:]): # [:5] means downloard only first 5 samples, if it works, please remove it
    print(idx + 41093)
    img = i.split('_')
    for color in colors:
        img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
        img_name = i + "_" + color + ".jpg"
        img_url = v18_url + img_path
        try:
            r = requests.get(img_url, allow_redirects=True)
        except requests.Timeout as e:
            print(e)
            continue
        print(color)
        #r = requests.get(img_url, allow_redirects=True)
        open(DIR + img_name, 'wb').write(r.content)
"""
import os
import errno
from multiprocessing.pool import Pool
from tqdm import tqdm
import requests
import pandas as pd
from PIL import Image

def download(pid, image_list, base_url, save_dir, image_size=(512, 512)):
    colors = ['red', 'green', 'blue', 'yellow']
    try:
        for i in tqdm(image_list, postfix={"pid": pid}):
            img_id = i.split('_', 1)
            for color in colors:
                img_path = img_id[0] + '/' + img_id[1] + '_' + color + '.jpg'
                img_name = i + '_' + color + '.png'
                if os.path.exists(os.path.join(save_dir, img_name)):
                    continue
                img_url = base_url + img_path

                # Get the raw response from the url
                r = requests.get(img_url, allow_redirects=True, stream=True)
                r.raw.decode_content = True

                # Use PIL to resize the image and to convert it to L
                # (8-bit pixels, black and white)
                im = Image.open(r.raw)
                im = im.resize(image_size, Image.LANCZOS)
                #im = im.convert('L')
                im.save(os.path.join(save_dir, img_name), 'PNG')
    except Exception as e:
        print(e)

if __name__ == '__main__':
    # Parameters
    process_num = 24
    image_size = (512, 512)
    url = 'http://v18.proteinatlas.org/images/'
    csv_path =  "./HPAv18RBGY_wodpl.csv"
    save_dir = "./HPAv18/512_RGB"

    # Create the directory to save the images in case it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print('Parent process %s.' % os.getpid())
    img_list = pd.read_csv(csv_path)['Id']
    list_len = len(img_list)
    p = Pool(process_num)
    for i in range(process_num):
        start = int(i * list_len / process_num)
        end = int((i + 1) * list_len / process_num)
        process_images = img_list[start:end]
        p.apply_async(
            download, args=(str(i), process_images, url, save_dir, image_size)
        )
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
