import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd

from config.config import *
from utils.common_util import *

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    # https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72534
    test_leak_df = pd.read_csv(opj(DATA_DIR, 'raw', 'TestEtraMatchingUnder_259_R14_G12_B10.csv'), usecols=['Test', 'Extra'])
    test_leak_df['Extra'] = test_leak_df['Extra'].apply(lambda x: x[x.find('_') + 1:])
    test_leak_df = test_leak_df.rename(columns={'Test': ID})
    test_leak_df[EXTERNAL] = False

    external_meta_df = pd.read_csv(opj(DATA_DIR, 'meta', 'external_meta.csv'))
    external_meta_df = external_meta_df.rename(columns={ID: 'Extra'})

    test_leak_df = pd.merge(test_leak_df, external_meta_df[['Extra', TARGET, ANTIBODY, ANTIBODY_CODE]], on='Extra', how='left')
    test_leak_df = test_leak_df[[ID, EXTERNAL, TARGET, ANTIBODY, ANTIBODY_CODE]].drop_duplicates(subset=ID)
    assert np.all(test_leak_df[TARGET].notnull())

    print(test_leak_df.shape)
    print(test_leak_df.head())
    test_leak_df.to_csv(opj(DATA_DIR, 'meta/test_leak_meta.csv'), index=False)

    print('\nsuccess!')
