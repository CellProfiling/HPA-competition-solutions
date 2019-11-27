import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd

from config.config import *
from utils.common_util import *

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    split_df = pd.read_csv(opj(DATA_DIR, 'split/external_antibody_split.csv'))
    base_cols = split_df.columns.values.tolist()

    train_extra = pd.read_csv(opj(DATA_DIR, 'raw', 'train_extra.csv'))
    train_extra[ID] = train_extra[ID].apply(lambda x: x.replace('/', '_'))
    train_extra = train_extra.drop_duplicates().reset_index(drop=True)

    split_df = pd.merge(split_df, train_extra, on=ID, how='left', suffixes=('', '_new'))
    target1 = split_df[TARGET].values
    target2 = split_df['%s_new' % TARGET].values

    target_list = []
    exclude_tags = ['Peroxisomes', 'Endosomes', 'Lysosomes', 'Lipid droplets', 'Cytoplasmic bodies']
    exclude_idxes = [LABEL_NAME_LIST.index(e) for e in exclude_tags]
    for t1, t2 in zip(target1, target2):
        t1 = np.array(t1.split(' '), dtype='int')
        if isinstance(t2, str):
            t2 = np.array(t2.split(' '), dtype='int')
            for idx in exclude_idxes:
                if idx in t1 and idx not in t2:
                    t2 = np.union1d(t2, [idx])
                if idx not in t1 and idx in t2:
                    t2 = np.setdiff1d(t2, [idx])
        else:
            t2 = []
        if len(t2) == 0:
            t2 = t1
        target_list.append(' '.join(np.sort(t2).astype(str).tolist()))
    target2 = target_list

    split_df[TARGET] = target2
    split_df = split_df[base_cols]
    split_df.to_csv(opj(DATA_DIR, 'meta/external_antibody_correct_meta.csv'), index=False)

    print('\nsuccess!')
