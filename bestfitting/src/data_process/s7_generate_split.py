import sys
sys.path.insert(0, '..')
import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from config.config import *
import cv2
from utils.common_util import *
from sklearn.model_selection import KFold,StratifiedKFold
# https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/67819
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def get_meta():
    meta_dir = opj(DATA_DIR,'meta')

    train_meta_fname = opj(meta_dir, 'train_meta.csv')
    train_meta = pd.read_csv(train_meta_fname)

    external_meta_fname = opj(meta_dir, 'external_meta.csv')
    external_meta = pd.read_csv(opj(external_meta_fname))
    return train_meta, external_meta


def create_split_file(data_set="train", name="train", num=None):
    split_dir = opj(DATA_DIR, 'split')
    os.makedirs(split_dir, exist_ok=True)

    ds = train_meta if data_set=='train' else pd.read_csv(opj(DATA_DIR, 'raw', 'sample_submission.csv'))
    if num is None:
        split_df = ds
    elif name == "valid":
        split_df = ds.iloc[-num:].copy()
    else:
        split_df = ds.iloc[:num]
    num = len(split_df)
    print("create split file: %s_%d" % (name, num))
    fname = opj(split_dir, "%s_%d.csv" % (name, num))
    split_df.to_csv(fname, index=False)


def create_random_split(train_meta, external_meta=None, n_splits=4, alias='random'):
    split_dir = opj(DATA_DIR, 'split', '%s_folds%d' % (alias, n_splits))
    os.makedirs(split_dir, exist_ok=True)

    kf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=100)
    train_indices_list, valid_indices_list = [], []
    for train_indices, valid_indices in kf.split(train_meta, train_meta[LABEL_NAME_LIST].values):
        train_indices_list.append(train_indices)
        valid_indices_list.append(valid_indices)

    ext_train_indices_list, ext_valid_indices_list = [], []
    if external_meta is not None:
        ext_kf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=100)
        for ext_train_indices, ext_valid_indices in ext_kf.split(external_meta, external_meta[LABEL_NAME_LIST].values):
            ext_train_indices_list.append(ext_train_indices)
            ext_valid_indices_list.append(ext_valid_indices)

    for idx in range(n_splits):
        train_split_df = train_meta.loc[train_indices_list[idx]]
        valid_split_df = train_meta.loc[valid_indices_list[idx]]

        if external_meta is not None:
            train_split_df = pd.concat((train_split_df,
                                        external_meta.loc[ext_train_indices_list[idx]]), ignore_index=True)
            valid_split_df = pd.concat((valid_split_df,
                                        external_meta.loc[ext_valid_indices_list[idx]]), ignore_index=True)
            train_split_df = train_split_df[[ID, TARGET, EXTERNAL, ANTIBODY, ANTIBODY_CODE] + LABEL_NAME_LIST]
            valid_split_df = valid_split_df[[ID, TARGET, EXTERNAL, ANTIBODY, ANTIBODY_CODE] + LABEL_NAME_LIST]

        if idx == 0:
            for name in LABEL_NAMES.values():
                print(name, (train_split_df[name]==1).sum(), (valid_split_df[name]==1).sum())

        fname = opj(split_dir, 'random_train_cv%d.csv' % (idx))
        print("create split file: %s, shape: %s" % (fname, str(train_split_df.shape)))
        train_split_df.to_csv(fname, index=False)

        fname = opj(split_dir, 'random_valid_cv%d.csv' % (idx))
        print("create split file: %s, shape: %s" % (fname, str(valid_split_df.shape)))
        valid_split_df.to_csv(fname, index=False)


def load_match_info():
    train_meta_df = pd.read_csv(opj(DATA_DIR, 'meta', 'train_meta.csv'))
    external_meta_df = pd.read_csv(opj(DATA_DIR, 'meta', 'external_meta.csv'))
    meta_df = pd.concat((train_meta_df[[ID, TARGET]], external_meta_df[[ID, TARGET]]), axis=0, ignore_index=True)

    match_df = pd.read_csv(opj(DATA_DIR, 'meta', 'train_match_external.csv.gz'), usecols=['Train', 'Extra'])
    match_df = pd.merge(match_df, meta_df.rename(columns={ID: 'Train'}), on='Train', how='left')
    match_df = match_df.rename(columns={TARGET: 'Train_%s' % TARGET})
    match_df = pd.merge(match_df, meta_df.rename(columns={ID: 'Extra'}), on='Extra', how='left')
    match_df = match_df.rename(columns={TARGET: 'Extra_%s' % TARGET})

    match_df['Equal'] = match_df['Train_%s' % TARGET] == match_df['Extra_%s' % TARGET]
    match_df = match_df[['Train', 'Extra', 'Train_%s' % TARGET, 'Extra_%s' % TARGET, 'Equal']]
    return match_df


def generate_noleak_split(n_splits=5):
    match_df = load_match_info()
    match_img_ids = match_df[~match_df['Equal']]['Extra'].unique()

    target_dir = opj(DATA_DIR, 'split', 'random_ext_noleak_clean_folds5')
    os.makedirs(target_dir, exist_ok=True)
    for idx in range(n_splits):
        train_split_df = pd.read_csv(opj(DATA_DIR, 'split', 'random_ext_folds5', 'random_train_cv%d.csv' % idx))
        start_num = len(train_split_df)
        train_split_df = train_split_df[~train_split_df[ID].isin(match_img_ids)]
        end_num = len(train_split_df)
        print('trainset remove num: %d' % (start_num - end_num))

        valid_split_df = pd.read_csv(opj(DATA_DIR, 'split', 'random_ext_folds5', 'random_valid_cv%d.csv' % idx))
        start_num = len(valid_split_df)
        leak_img_ids = match_df[(match_df['Equal']) & (match_df['Train'].isin(train_split_df[ID].values))]['Extra'].unique()
        valid_split_df = valid_split_df[(~valid_split_df[ID].isin(match_img_ids)) & (~valid_split_df[ID].isin(leak_img_ids))]
        end_num = len(valid_split_df)
        print('validset remove num: %d' % (start_num - end_num))

        fname = opj(target_dir, 'random_train_cv%d.csv' % idx)
        print(fname, train_split_df.shape)
        train_split_df.to_csv(fname, index=False)

        fname = opj(target_dir, 'random_valid_cv%d.csv' % idx)
        print(fname, valid_split_df.shape)
        valid_split_df.to_csv(fname, index=False)


parser = argparse.ArgumentParser(description='create split')
args = parser.parse_args()

if __name__ == "__main__":
    print('%s: calling main function ... ' % os.path.basename(__file__))

    train_meta, external_meta = get_meta()

    create_split_file(data_set="train", name="train", num=160)
    create_split_file(data_set="train", name="valid", num=160)
    create_split_file(data_set="test", name="test", num=160)

    create_split_file(data_set="train", name="train", num=None)
    create_split_file(data_set="test", name="test", num=None)

    create_random_split(train_meta, n_splits=5)
    create_random_split(train_meta, external_meta=external_meta, n_splits=5, alias='random_ext')

    generate_noleak_split(n_splits=5)

    print('\nsuccess!')
