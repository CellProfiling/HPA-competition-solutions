import sys
sys.path.insert(0, '..')
from config.config import *
from utils.common_util import *
from utils.leak_util import *

import numpy as np
import pandas as pd

def sort_targets(x):
    if x is None or x == '':
        return x

    x = x.split(' ')
    x = np.array(x, dtype=int)
    x = np.sort(x)
    x = x.astype(str).tolist()
    x = ' '.join(x)
    return x

def fill_targets(row):
    row.Predicted = np.array(row.Predicted.split(" ")).astype(np.int)
    for num in row.Predicted:
        name = LABEL_NAMES[int(num)]
        row.loc[name] = 1
    return row

def show_modify_info(before_df, after_df):
    before_df = before_df.copy()
    after_df = after_df.copy()

    for label in LABEL_NAME_LIST:
        before_df[label] = 0
    before_df = before_df.apply(fill_targets, axis=1)
    before_ts = before_df[LABEL_NAME_LIST].sum()

    for label in LABEL_NAME_LIST:
        after_df[label] = 0
    after_df = after_df.apply(fill_targets, axis=1)
    after_ts = after_df[LABEL_NAME_LIST].sum()

    assert np.all(before_df[ID].values == after_df[ID].values)
    df = after_df[LABEL_NAME_LIST] - before_df[LABEL_NAME_LIST]

    increase_ts = (df == 1).sum()
    decrease_ts = -(df == -1).sum()

    info_df = pd.concat((before_ts, after_ts, increase_ts, decrease_ts), axis=1)
    info_df.columns = ['before', 'after', 'increase', 'decrease']
    assert np.all((info_df['after'] - info_df['before']).values == (info_df['increase'] + info_df['decrease']).values)
    info_df['modify'] = info_df['after'] - info_df['before']
    print(info_df)

def modify_match_target(match_df):
    train_extra = pd.read_csv(opj(DATA_DIR, 'raw', 'train_extra.csv'))
    train_extra[ID] = train_extra[ID].apply(lambda x: x.replace('/', '_'))
    train_extra = train_extra.drop_duplicates().reset_index(drop=True)

    match_df = pd.merge(match_df, train_extra, left_on='train_id', right_on=ID, how='left', suffixes=('', '_truth'))

    idxes = match_df['%s_truth' % TARGET].notnull()
    target1 = match_df.loc[idxes, 'train_target'].values
    target2 = match_df.loc[idxes, '%s_truth' % TARGET].values

    target_list = []
    exclude_tags = ['Peroxisomes', 'Endosomes', 'Lysosomes', 'Lipid droplets', 'Cytoplasmic bodies']
    exclude_idxes = [LABEL_NAME_LIST.index(e) for e in exclude_tags]
    for t1, t2 in zip(target1, target2):
        t1 = np.array(t1.split(' '), dtype='int')
        t2 = np.array(t2.split(' '), dtype='int')
        for idx in exclude_idxes:
            if idx in t1 and idx not in t2:
                t2 = np.union1d(t2, [idx])
            if idx not in t1 and idx in t2:
                t2 = np.setdiff1d(t2, [idx])
        if len(t2) == 0:
            t2 = t1
        target_list.append(' '.join(np.sort(t2).astype(str).tolist()))
    target2 = target_list

    match_df.loc[idxes, 'train_target'] = target2

    return match_df

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_name', default='external_crop1024_focal_slov_hardlog_clean_class_densenet121_large_dropout_i1536_aug2_5folds',
                    type=str, help='model_name')
parser.add_argument('--face_model_name', default='face_all_class_resnet50_dropout_i768_aug2_5folds',
                    type=str, help='face_model_name')
parser.add_argument('--out_name', default='d121_i1536_aug2_maximum_5folds_f012_max_test_ratio2_face_r50_i768',
                    type=str, help='alias_name')
parser.add_argument('--threshold', default=0.65, type=float, help='')
args = parser.parse_args()

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    model_name = args.model_name
    face_model_name = args.face_model_name
    out_name = args.out_name
    threshold = args.threshold
    dataset = 'test'

    sub_dir = opj(RESULT_DIR, 'submissions/ensemble', model_name)

    print('load dir: %s' % sub_dir)
    base_df = pd.read_csv(opj(sub_dir, 'results_test.csv.gz'))

    # --------------------------- start --------------------------- #
    base_df[PREDICTED] = base_df[PREDICTED].apply(lambda x: sort_targets(x))
    df = base_df.copy()
    match_df = pd.read_csv(opj(RESULT_DIR, 'cache/match', face_model_name, '%s_match_th%.2f.csv' % (dataset, threshold)))
    match_df = modify_match_target(match_df)

    print('-%d-' % len(match_df))

    for idx in range(len(match_df)):
        id_value, target = match_df.iloc[idx][[ID, 'train_target']].values
        df.loc[df[ID] == id_value, PREDICTED] = target
    df[PREDICTED] = df[PREDICTED].apply(lambda x: sort_targets(x))
    assert np.all(base_df[ID].values == df[ID].values)
    print('modify num: %d' % (base_df[PREDICTED].values != df[PREDICTED].values).sum())

    show_modify_info(base_df, df)
    # ---------------------------- end ---------------------------- #

    fname = opj(sub_dir, '%s_%d.csv.gz' % (out_name, len(match_df)))
    print('out file: %s' % fname)
    df.to_csv(fname, index=False, compression='gzip')

    print('\nsuccess!')
