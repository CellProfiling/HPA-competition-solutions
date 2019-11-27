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

def test_imread(img_dir, img_id, color):
    img = Image.open(opj(img_dir, '%s_%s.png' % (img_id, color)))
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
    color, threshold, base_test_hash1, base_test_hash2, test_ids1, test_ids2 = params

    test_hash1 = base_test_hash1.reshape(1, -1)  # 1*m

    test_idxes_list1 = []
    test_idxes_list2 = []
    hash_list = []

    step = 5
    for test_idx in tqdm(range(0, len(base_test_hash2), step), desc=color):
        test_hash2 = base_test_hash2[test_idx:test_idx + step].reshape(-1, 1)  # n*1
        hash = test_hash2 - test_hash1  # n*m
        test_idxes2, test_idxes1 = np.where(hash <= threshold)
        hash = hash[test_idxes2, test_idxes1]

        test_idxes2 = test_idxes2 + test_idx

        test_idxes_list1.extend(test_idxes1.tolist())
        test_idxes_list2.extend(test_idxes2.tolist())
        hash_list.extend(hash.tolist())

    df = pd.DataFrame({
        'Test1': test_ids1[test_idxes_list1],
        'Test2': test_ids2[test_idxes_list2],
        'Sim%s' % color[:1].upper(): hash_list
    })
    df = df[df['Test1'] != df['Test2']]
    return df

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    # test set images
    test_img_dir = opj(DATA_DIR, 'test', 'images')
    test_meta = pd.read_csv(opj(DATA_DIR, 'raw', 'sample_submission.csv'))

    colors = ['red', 'green', 'blue']
    test_meta = generate_hash(test_img_dir, test_meta, colors,
                              dataset='test', imread_func=test_imread, is_update=False)

    threshold = 12

    pool = mlc.SuperPool(3)
    params = []
    for color in colors:
        base_test_hash1 = test_meta[color].values
        base_test_hash2 = test_meta[color].values

        test_ids1 = test_meta[ID].values
        test_ids2 = test_meta[ID].values

        params.append((color, threshold, base_test_hash1, base_test_hash2, test_ids1, test_ids2))
    df_list = pool.map(calc_hash, params)

    df = None
    for temp_df, color in zip(df_list, colors):
        if df is None:
            df = temp_df
        else:
            df = pd.merge(df, temp_df, on=['Test1', 'Test2'], how='inner')

    print(df.shape)
    df.to_csv(opj(DATA_DIR, 'meta', 'test_match_test.csv.gz'), index=False, compression='gzip')

    print('\nsuccess!')
