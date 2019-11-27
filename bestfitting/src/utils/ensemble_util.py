import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config.config import *
from utils.common_util import *
from layers.kaggle_metric import fit_value_th, fit_test_th
from layers.loss import *

def search_val_threshold(probs, meta_df, to_dir=None, sub_data_type=None, update=False):
    if sub_data_type is None:
        threshold_fname = opj(to_dir, 'val_threshold.npy')
    else:
        threshold_fname = opj(to_dir, '%s_val_threshold.npy' % sub_data_type)
    if ope(threshold_fname) and not update:
        threshold = np.load(threshold_fname)
    else:
        labels = meta_df[LABEL_NAME_LIST].values
        assert probs.shape == labels.shape

        print('search threshold, wait a moment...')

        threshold = fit_value_th(probs, labels)
        if to_dir is None:
            np.save(threshold_fname, threshold)
    return threshold

def search_test_threshold(probs, ratio, to_dir=None, sub_data_type=None, update=False):
    print(ratio)

    if sub_data_type is None:
        threshold_fname = opj(str(to_dir), 'test_threshold.npy')
    else:
        threshold_fname = opj(str(to_dir), '%s_test_threshold.npy' % sub_data_type)
    if ope(threshold_fname) and not update:
        threshold = np.load(threshold_fname)
    else:
        print('search threshold, wait a moment...')

        threshold = fit_test_th(probs, ratio)
        assert ratio.shape == threshold.shape
        if to_dir is not None:
            np.save(threshold_fname, threshold)
    return threshold

def generate_labels(probs, threshold=0.5):
    labels = []
    for prob in probs:
        prob = prob.copy()
        prob[np.argmax(prob)] = 1
        labels.append(' '.join(np.where(prob > threshold)[0].astype(str).tolist()))
    return labels

def generate_submit(df, probs, threshold=0.5):
    assert len(df) == len(probs)
    labels = generate_labels(probs, threshold=threshold)
    df[PREDICTED] = labels
    return df

def fill_targets(row):
    row.Predicted = np.array(row.Predicted.split(" ")).astype(np.int)
    for num in row.Predicted:
        name = LABEL_NAMES[int(num)]
        row.loc[name] = 1
    return row

def get_kaggle_score(result_df, train_df, base_probs):
    result_df = result_df.copy()

    for key in LABEL_NAMES.keys():
        result_df[LABEL_NAMES[key]] = 0
    result_df = result_df.apply(fill_targets, axis=1)

    sub_train_df = train_df[train_df[ID].isin(result_df[ID].values)]
    labels = sub_train_df[LABEL_NAME_LIST].values

    sub_result_df = pd.merge(sub_train_df[[ID]], result_df, on=ID, how='left')
    preds = sub_result_df[LABEL_NAME_LIST].values

    acc = (labels == preds).mean()
    print('valid acc: %.5f' % acc)

    score = f1_score(labels, preds, average='macro')
    print('valid kaggle score: %.5f' % score)

    logit = -np.log(1 / base_probs - 1)
    logit = torch.from_numpy(logit).float()
    labels = torch.from_numpy(labels).float()
    focal_loss = FocalLoss().forward(logit, labels)
    print('valid focal loss: %.5f' % focal_loss)
    return acc, score, focal_loss

def generate_distribution(df):
    df = df.copy()
    for key in LABEL_NAMES.keys():
        df[LABEL_NAMES[key]] = 0
    df = df.apply(fill_targets, axis=1)
    return df[LABEL_NAME_LIST].values.sum(axis=0)
