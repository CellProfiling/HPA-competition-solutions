import sys
sys.path.insert(0, '..')
import requests
import pandas as pd

from tqdm import tqdm
import mlcrate as mlc

from config.config import *
from utils.common_util import *

# https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69984
def generate_imgs(param):
    img_id, v18_url, external_dir = param
    img_id_list = img_id.split('_')

    for color in ['red', 'green', 'blue', 'yellow']:
        img_url = v18_url + img_id_list[0] + '/' + '_'.join(img_id_list[1:]) + '_' + color + '.jpg'
        img_name = img_id + '_' + color + '.jpg'

        fpath = opj(external_dir, img_name)
        if not ope(fpath):
            r = requests.get(img_url, allow_redirects=True)
            open(fpath, 'wb').write(r.content)

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    external_dir = opj(EXTERNEL_DIR, 'HPAv18', 'jpg')
    v18_url = 'http://v18.proteinatlas.org/images/'

    os.makedirs(external_dir, exist_ok=True)
    img_list = pd.read_csv(opj(DATA_DIR, 'raw', 'HPAv18RBGY_wodpl.csv'))

    params = [(img_id, v18_url, external_dir) for img_id in img_list['Id'].values]

    pool = mlc.SuperPool(8)
    pool.map(generate_imgs, params, description='download hpa v18')

    print('\nsuccess!')
