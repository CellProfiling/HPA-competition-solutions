import sys
sys.path.insert(0, '..')
import os
import gc

import numpy as np
import pandas as pd
from config.config import *
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from timeit import default_timer as timer

opj = os.path.join
ope = os.path.exists


def load_data(dataset='train'):
    feature_fname = opj(model_dir, 'extract_feats_%s.npz' % dataset)

    X = np.load(feature_fname)
    features = X['feats']
    df = pd.DataFrame({ID: X['ids']})
    if debug:
        num = 2000
        df = df[:num]
        features = features[:num]

    if dataset == 'train':
        meta_df = pd.read_csv(opj(DATA_DIR, 'split/external_trainset_antibody_split.csv'))
    elif dataset == 'val':
        meta_df = pd.read_csv(opj(DATA_DIR, 'split/external_validset_antibody_split.csv'))
    elif dataset == 'ext':
        meta_df = pd.read_csv(opj(DATA_DIR, 'meta/external_antibody_correct_meta.csv'))
    elif dataset == 'test':
        meta_df = pd.read_csv(opj(DATA_DIR, 'meta/test_leak_meta.csv'))
    else:
        raise ValueError(dataset)

    df = pd.merge(df, meta_df, on=ID, how='left')
    print('dataset %s, num: %d' % (dataset, len(df)))

    data = (df, features)
    return data


def cosin_metric(x1, x2):
    '''
    :param x1: (m, k)
    :param x2: (n, k)
    :return: (m, n)
    '''
    x1 = preprocessing.scale(x1)
    x2 = preprocessing.scale(x2)

    assert x1.shape[-1] == x2.shape[-1]

    norm_x1 = np.linalg.norm(x1, axis=1).reshape(-1, 1) # (m, 1)
    norm_x2 = np.linalg.norm(x2, axis=1).reshape(1, -1) # (1, n)

    return np.dot(x1, x2.T) / norm_x1 / norm_x2


def generate_label(y1, y2):
    '''
    :param y1: (m,)
    :param y2: (n,)
    :return: (m, n)
    '''
    y1 = y1.reshape(-1, 1) # (m, 1)
    y2 = y2.reshape(1, -1) # (1, n)

    label1 = np.ones_like(y2) # (1, n)
    label1 = y1 * label1 # (m, n)

    label2 = np.ones_like(y1) # (m, 1)
    label2 = label2 * y2 # (m, n)

    label = (label1 == label2).astype('uint8')
    return label


def save_top3_results(train_df, valid_df, cosin_dist, data_type):
    valid_ids = valid_df[ID].values
    top_index = [sub[::-1][:3] for sub in np.argsort(cosin_dist, axis=1)]
    top_data = []

    for index, valid_id in tqdm(enumerate(valid_ids)):
        data = [valid_id]
        top_ix = top_index[index]
        dists = list(cosin_dist[index][top_ix])
        match_ids = list(train_df.iloc[top_ix][ID].values)
        data.extend(match_ids)
        data.extend(dists)
        top_data.append(data)

    columns = [ID, 'top1', 'top2', 'top3', 'top1_score', 'top2_score', 'top3_score']
    df = pd.DataFrame(data=top_data, columns=columns)

    out_dir = opj(RESULT_DIR, 'cache', 'match', model_name)
    os.makedirs(out_dir, exist_ok=True)
    fname = opj(out_dir, '%s_top3.csv'%(data_type))
    df.to_csv(fname, index=False)


def do_match(data_type='val'):
    if data_type == 'val':
        train_data = load_data(dataset='train') # df, features
    else:
        train_data = load_data(dataset='ext') # df, features
    valid_data = load_data(dataset=data_type) # df, features

    train_features = train_data[-1]
    valid_features = valid_data[-1]
    cosin_dist = cosin_metric(valid_features, train_features)

    train_df = train_data[0]
    valid_df = valid_data[0]

    save_top3_results(train_df, valid_df, cosin_dist, data_type)

    train_label = train_df[ANTIBODY_CODE].values
    valid_label = valid_df[ANTIBODY_CODE].values
    label = generate_label(valid_label, train_label)
    match_max_num = (label.sum(axis=1)>0).sum()
    print('label match count', match_max_num)

    max_cosin = np.max(cosin_dist, axis=1)
    out_dir = opj(RESULT_DIR, 'cache', 'match', model_name)
    os.makedirs(out_dir, exist_ok=True)
    for threshold in np.arange(0.6, 0.8, 0.01):
        ix = max_cosin > threshold
        sel_cosin_dist = cosin_dist[ix]
        sel_label = label[ix]
        sel_valid_df = valid_df[ix].copy()

        sample_num = len(sel_label)
        argmax = np.argmax(sel_cosin_dist, axis=1)
        max_cosin_dist = np.max(sel_cosin_dist, axis=1)
        is_match = sel_label[range(sample_num), argmax]

        match_train_df = train_df.iloc[argmax]
        df = sel_valid_df[[ID, ANTIBODY_CODE, TARGET]]
        df['train_id'] = match_train_df[ID].values
        df['train_antibody_code'] = match_train_df[ANTIBODY_CODE].values
        df['train_antibody'] = match_train_df[ANTIBODY].values
        df['train_target'] = match_train_df[TARGET].values

        is_correct = df[TARGET]==df['train_target']
        df['is_match'] = is_match
        df['is_correct'] = is_correct
        df['cosin_dist'] = max_cosin_dist

        match_num = np.sum(is_match)
        correct_num = np.sum(is_correct)
        acc = correct_num/(sample_num+EPS)
        recall = correct_num / len(valid_df)
        f1 = acc * recall * 2 / (acc + recall + EPS)
        print('threshold:%.2f, count:%d, match_num:%d, correct_num:%d, f1:%.4f acc:%.4f' % (threshold, sample_num, match_num, correct_num, f1, acc))
        fname = opj(out_dir, '%s_match_th%.2f.csv'%(data_type, threshold))
        df.to_csv(fname, index=False)


import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_name', default='face_all_class_resnet50_dropout_i768_aug2_5folds', type=str, help='model_name')
parser.add_argument('--epoch_name', default='045', type=str, help='cfg name')
parser.add_argument('--debug', default=0, type=int, help='cfg name')
parser.add_argument('--do_valid', default=1, type=int, help='')
parser.add_argument('--do_test', default=1, type=int, help='')

args = parser.parse_args()

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    start = timer()

    model_name = args.model_name
    epoch_name = args.epoch_name
    epoch_name = 'epoch_%s' % epoch_name
    debug = args.debug == 1
    do_valid = args.do_valid == 1
    do_test = args.do_test == 1
    target_dict = {}

    model_dir = opj(RESULT_DIR, 'submissions', model_name, epoch_name)
    if do_valid:
        do_match(data_type='val')
    if do_test:
        do_match(data_type='test')

    end = timer()
    time0 = (end - start) / 60
    print('Time spent for cluster: %3.1f min' % time0)

    print('\nsuccess!')
