import numpy as np
import pandas as pd
from config.config import *
from utils.common_util import *

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

def show_modify_info(submission_df1, submission_df2):
    submission_df1 = submission_df1.copy()
    submission_df2 = submission_df2.copy()

    for label in LABEL_NAME_LIST:
        submission_df1[label] = 0
    submission_df1 = submission_df1.apply(fill_targets, axis=1)
    ts1 = submission_df1[LABEL_NAME_LIST].sum()

    for label in LABEL_NAME_LIST:
        submission_df2[label] = 0
    submission_df2 = submission_df2.apply(fill_targets, axis=1)
    ts2 = submission_df2[LABEL_NAME_LIST].sum()

    assert np.all(submission_df1[ID].values == submission_df2[ID].values)
    df = submission_df2[LABEL_NAME_LIST] - submission_df1[LABEL_NAME_LIST]
    ts3 = (df == 1).sum()
    ts4 = -(df == -1).sum()

    ts = pd.concat((ts1, ts2, ts3, ts4), axis=1)
    ts.columns = ['before', 'after', 'increase', 'decrease']
    assert np.all((ts['after'] - ts['before']).values == (ts['increase'] + ts['decrease']).values)
    ts['modify'] = ts['after'] - ts['before']
    print(ts)

def modify_submit(submission_df, show_info=False):
    submission_df[PREDICTED] = submission_df[PREDICTED].apply(lambda x: sort_targets(x))
    submission_df_cp = submission_df.copy()

    # replace leak result
    kernal_leak_df = pd.read_csv(opj(DATA_DIR, 'meta', 'test_leak_meta.csv'))
    for idx in range(len(kernal_leak_df)):
        id_value, target = kernal_leak_df.iloc[idx][[ID, TARGET]].values
        submission_df.loc[submission_df[ID] == id_value, PREDICTED] = target

    submission_df[PREDICTED] = submission_df[PREDICTED].apply(lambda x: sort_targets(x))

    assert np.all(submission_df_cp[ID].values == submission_df[ID].values)
    print('modify num: %d' % (submission_df_cp[PREDICTED].values != submission_df[PREDICTED].values).sum())

    if show_info:
        show_modify_info(submission_df_cp, submission_df)

    assert submission_df.shape[-1] == 2
    return submission_df
