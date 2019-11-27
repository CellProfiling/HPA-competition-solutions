import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config.config import *
from utils.common_util import *

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = LABEL_NAMES[int(num)]
        row.loc[name] = 1
    row.Target = ' '.join(np.sort(np.unique(row.Target)).astype(str).tolist())
    return row

def generate_meta(meta_dir, fname, dataset='train'):
    is_external = True if dataset == 'external' else False

    label_df = pd.read_csv(opj(DATA_DIR, 'raw', fname))
    for key in LABEL_NAMES.keys():
        label_df[LABEL_NAMES[key]] = 0
    meta_df = label_df.apply(fill_targets, axis=1)
    meta_df[EXTERNAL] = is_external

    if is_external:
        meta_df[ANTIBODY] = meta_df[ID].apply(lambda x: x.split('_')[0])

        clf = LabelEncoder()
        meta_df[ANTIBODY_CODE] = clf.fit_transform(meta_df[ANTIBODY])
        meta_df[ANTIBODY] = meta_df[ANTIBODY].astype(int)

    meta_fname = opj(meta_dir, '%s_meta.csv' % dataset)
    meta_df.to_csv(meta_fname, index=False)

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    meta_dir = opj(DATA_DIR, 'meta')
    os.makedirs(meta_dir, exist_ok=True)

    generate_meta(meta_dir, 'train.csv', dataset='train')

    # https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69984
    generate_meta(meta_dir, 'HPAv18RBGY_wodpl.csv', dataset='external')

    print('\nsuccess!')
