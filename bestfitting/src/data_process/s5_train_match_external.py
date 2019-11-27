import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import imagehash
from tqdm import tqdm
import pickle
import mlcrate as mlc

from config.config import *
from utils.common_util import *

def train_imread(img_dir, img_id, color):
    img = Image.open(opj(img_dir, '%s_%s.png' % (img_id, color)))
    return img

def external_imread(img_dir, img_id, color):
    img = cv2.imread(opj(img_dir, '%s_%s.jpg' % (img_id, color)), cv2.IMREAD_GRAYSCALE)
    img = Image.fromarray(img)
    return img

# https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72534
def generate_hash(img_dir, meta, colors, dataset='train', imread_func=None, is_update=False):
    meta = meta.copy()
    cache_fname = opj(DATA_DIR, 'meta', '%s_hash_maps.pkl' % dataset)
    if ope(cache_fname) and not is_update:
        with open(cache_fname, 'rb') as dbfile:
            hash_maps = pickle.load(dbfile)
    else:
        hash_maps = {}
        for color in colors:
            hash_maps[color] = []
            for idx in tqdm(range(len(meta)), desc='train %s' % color):
                img = imread_func(img_dir, meta.iloc[idx][ID], color)
                hash = imagehash.phash(img)
                hash_maps[color].append(hash)

        with open(cache_fname, 'wb') as dbfile:
            pickle.dump(hash_maps, dbfile)

    for color in colors:
        meta[color] = hash_maps[color]

    return meta

def calc_hash(params):
    color, th, base_external_hash, base_train_hash, train_ids, external_ids = params

    external_hash = base_external_hash.reshape(1, -1)  # 1*m

    train_idxes_list = []
    external_idxes_list = []
    hash_list = []

    step = 5
    for train_idx in tqdm(range(0, len(base_train_hash), step), desc=color):
        train_hash = base_train_hash[train_idx:train_idx + step].reshape(-1, 1)  # n*1
        hash = train_hash - external_hash  # n*m
        train_idxes, external_idxes = np.where(hash <= th)
        hash = hash[train_idxes, external_idxes]

        train_idxes = train_idxes + train_idx

        train_idxes_list.extend(train_idxes.tolist())
        external_idxes_list.extend(external_idxes.tolist())
        hash_list.extend(hash.tolist())

    df = pd.DataFrame({
        'Train': train_ids[train_idxes_list],
        'Extra': external_ids[external_idxes_list],
        'Sim%s' % color[:1].upper(): hash_list
    })
    return df

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    threshold = 12

    # train set images
    train_img_dir = opj(DATA_DIR, 'train', 'images')
    train_meta = pd.read_csv(opj(DATA_DIR, 'meta', 'train_meta.csv'))

    # external images
    external_img_dir = opj(DATA_DIR, 'train', 'external_v18_512')
    external_meta = pd.read_csv(opj(DATA_DIR, 'meta', 'external_meta.csv'))

    colors = ['red', 'green', 'blue']
    train_meta = generate_hash(train_img_dir, train_meta, colors,
                               dataset='train', imread_func=train_imread, is_update=False)
    external_meta = generate_hash(external_img_dir, external_meta, colors,
                                  dataset='external', imread_func=external_imread, is_update=False)

    pool = mlc.SuperPool(3)
    params = []
    for color in colors:
        base_tran_hash = train_meta[color].values
        base_external_hash = external_meta[color].values

        train_ids = train_meta[ID].values
        external_ids = external_meta[ID].values

        params.append((color, threshold, base_external_hash, base_tran_hash, train_ids, external_ids))
    df_list = pool.map(calc_hash, params)

    df = None
    for temp_df, color in zip(df_list, colors):
        if df is None:
            df = temp_df
        else:
            df = pd.merge(df, temp_df, on=['Train', 'Extra'], how='inner')

    print(df.shape)
    df.to_csv(opj(DATA_DIR, 'meta', 'train_match_external.csv.gz'), index=False, compression='gzip')

    print('\nsuccess!')
