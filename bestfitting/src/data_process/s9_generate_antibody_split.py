import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config.config import *
from utils.common_util import *

def create_external_split(meta_df):
    fname = opj(DATA_DIR, 'split', 'external_antibody_split.csv')
    print(meta_df.shape)
    print(meta_df.head())
    print('out file: %s' % fname)
    meta_df.to_csv(fname, index=False)

def create_train_valid_split(meta_df):
    ratio = 1. / 4
    train_split_df_list = []
    valid_split_df_list = []
    for antibody in tqdm(meta_df['antibody'].unique()):
        sub_meta_df = meta_df[meta_df['antibody'] == antibody]

        size = int(len(sub_meta_df) * ratio)
        if size > 0:
            train_idxes, valid_idxes = train_test_split(np.arange(len(sub_meta_df)), test_size=size, random_state=100)
            train_split_df = sub_meta_df.iloc[train_idxes]
            valid_split_df = sub_meta_df.iloc[valid_idxes]
        else:
            train_split_df = sub_meta_df
            valid_split_df = pd.DataFrame()

        train_split_df_list.append(train_split_df)
        valid_split_df_list.append(valid_split_df)

    train_split_df = pd.concat(train_split_df_list, axis=0, ignore_index=True)
    valid_split_df = pd.concat(valid_split_df_list, axis=0, ignore_index=True)
    print(train_split_df.shape, valid_split_df.shape)
    print(train_split_df['antibody'].nunique(), valid_split_df['antibody'].nunique())
    print(len(np.setdiff1d(valid_split_df['antibody'].unique(), train_split_df['antibody'].unique())))

    fname = opj(DATA_DIR, 'split', 'external_trainset_antibody_split.csv')
    print('out file: %s' % fname)
    train_split_df.to_csv(fname, index=False)

    fname = opj(DATA_DIR, 'split', 'external_validset_antibody_split.csv')
    print('out file: %s' % fname)
    valid_split_df.to_csv(fname, index=False)

    return train_split_df, valid_split_df

def create_train_match_split(external_trainset_df, external_validset_df):
    base_cols = external_trainset_df.columns.values

    train_match_df = pd.read_csv(opj(DATA_DIR, 'meta/train_external_match.csv.gz'), usecols=['Train', 'Extra'])
    train_match_df = train_match_df.rename(columns={'Train': ID})
    train_match_df[ANTIBODY] = train_match_df['Extra'].apply(lambda x: int(x.split('_')[0]))

    antibody_df = external_trainset_df[[TARGET, ANTIBODY, ANTIBODY_CODE]].drop_duplicates()
    train_match_df = pd.merge(train_match_df, antibody_df, on=ANTIBODY, how='left')
    train_match_df[EXTERNAL] = False

    external_trainset_df = pd.concat((
        external_trainset_df[~external_trainset_df[ID].isin(train_match_df['Extra'].values)],
        train_match_df[base_cols].drop_duplicates(subset=ID),
    ), axis=0, ignore_index=True)

    external_validset_df = external_validset_df[~external_validset_df[ID].isin(train_match_df['Extra'].values)]

    train_match_img_ids = np.sort(train_match_df[ID].unique())
    print(len(train_match_img_ids))

    # Select 1000 samples as reserved samples,but I did not used them.
    np.random.seed(100)
    select_num = 1000
    train_match_img_ids_1k = np.sort(np.random.choice(train_match_img_ids, size=select_num, replace=False))

    before_num = len(external_trainset_df) + len(external_validset_df)
    external_trainset_df = external_trainset_df[~external_trainset_df[ID].isin(train_match_img_ids_1k)].reset_index(drop=True)
    external_validset_df = external_validset_df[~external_validset_df[ID].isin(train_match_img_ids_1k)].reset_index(drop=True)
    after_num = len(external_trainset_df) + len(external_validset_df)
    assert before_num - after_num == select_num

    # save split result
    external_trainset_df = external_trainset_df[base_cols]
    external_validset_df = external_validset_df[base_cols]
    assert len(np.intersect1d(external_trainset_df[ID].values, external_validset_df[ID].values)) == 0

    assert len(np.intersect1d(external_trainset_df[ID].values, train_match_img_ids_1k)) == 0
    assert len(np.intersect1d(external_validset_df[ID].values, train_match_img_ids_1k)) == 0

    assert len(np.intersect1d(external_trainset_df[ID].values, train_match_df['Extra'].values)) == 0
    assert len(np.intersect1d(external_validset_df[ID].values, train_match_df['Extra'].values)) == 0

    # (61147, 6) (12563, 6)
    print(external_trainset_df.shape, external_validset_df.shape)
    external_trainset_df.to_csv(opj(DATA_DIR, 'split/external_tr_trainset_antibody_split.csv'), index=False)
    external_validset_df.to_csv(opj(DATA_DIR, 'split/external_tr_validset_antibody_split.csv'), index=False)

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    meta_df = pd.read_csv(opj(DATA_DIR, 'meta', 'external_meta.csv'))
    meta_df = meta_df[[ID, EXTERNAL, TARGET, ANTIBODY, ANTIBODY_CODE]]
    meta_df = meta_df.drop_duplicates()
    print('external antibody nunique: %d' % meta_df[ANTIBODY].nunique())

    create_external_split(meta_df)

    external_trainset_df, external_validset_df = create_train_valid_split(meta_df)

    # Replace the 'leak' sample in V18 with the train set samples.
    create_train_match_split(external_trainset_df, external_validset_df)

    print('\nsuccess!')
