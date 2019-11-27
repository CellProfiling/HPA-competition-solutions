import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config.config import *

import os
import pandas as pd
import numpy as np
from config.config import *
from utils.common_util import *
from sklearn.metrics import f1_score, recall_score, precision_score
import scipy.optimize as opt


def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))


def F1_soft(preds,targs,th=0.5,d=50.0):
    preds = sigmoid_np(d*(preds - th))
    targs = targs.astype(np.float)
    score = 2.0*(preds*targs).sum(axis=0)/((preds+targs).sum(axis=0) + 1e-6)
    return score


def fit_value_th(x,y):
    params = 0.5*np.ones(len(LABEL_NAME_LIST))
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x,y,p) - 1.0, wd*(p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    p[p < 0.1] = 0.1
    return p

def fit_value_th_whole(x,y):
    params = np.array([0.5])
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x,y,p) - 1.0, wd*(p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    p[p < 0.1] = 0.1
    p = p[0]
    return p


def Count_soft(preds,th=0.5,d=50.0):
    preds = sigmoid_np(d*(preds - th))
    return preds.mean(axis=0)


def fit_test_th(x,y):
    p = []
    from tqdm import tqdm
    for idx in tqdm(range(len(y))):
        _y = y[idx]
        _x = x[:, idx]
        min_error = np.inf
        min_p = 0
        for _p in np.linspace(0, 1, 10000):
            error = np.abs((_x > _p).mean() - _y)
            if error < min_error:
                min_error = error
                min_p = _p
            elif error == min_error and (np.abs(_p - 0.5) < np.abs(min_p - 0.5)):
                min_error = error
                min_p = _p
        p.append(min_p)
    p = np.array(p)
    return p


def prob_to_result(probs, img_ids, th=0.5):
    probs = probs.copy()
    probs[range(len(probs)), np.argmax(probs, axis=1)] = 1

    pred_list = []
    for line in probs:
        s = ' '.join(list([str(i) for i in np.nonzero(line > th)[0]]))
        pred_list.append(s)
    df = pd.DataFrame({ID: img_ids, PREDICTED: pred_list})
    return df


def multi_class_acc(preds, targs, th=0.5):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()


def get_probs_f1_score(df, probs, truth, th=None):
    probs = probs.copy()
    probs[range(len(probs)), np.argmax(probs, axis=1)] = 1

    truth = df[[ID]].merge(truth, how='left', on=ID)
    labels = truth[LABEL_NAME_LIST].values.astype(int)
    assert labels.shape[1] == NUM_CLASSES

    if th is None:
        th = fit_value_th(probs, labels)

    pred = (probs>th).astype(int)
    score = f1_score(labels, pred, average='macro')
    return score


def get_probs_f1_score_perlabel(df, probs, th=None):
    probs = probs.copy()
    probs[range(len(probs)), np.argmax(probs, axis=1)] = 1

    truth = pd.read_csv(opj(DATA_DIR, 'meta', 'train_meta.csv'))
    truth = df[[ID]].merge(truth, how='left', on=ID)
    labels = truth[LABEL_NAME_LIST].values.astype(int)
    assert labels.shape[1] == NUM_CLASSES

    if th is None:
        th = fit_value_th(probs, labels)
    pred = (probs > th).astype(int)
    score = []
    for i in LABEL_NAMES.keys():
        f1 = f1_score(labels[:, i], pred[:, i])
        recall = recall_score(labels[:, i], pred[:, i])
        precision = precision_score(labels[:, i], pred[:, i])
        score.append([f1, recall, precision])
    score_df = pd.DataFrame(score, columns=['f1', 'recall', 'precision'])
    score_df.insert(0, 'Target', LABEL_NAME_LIST)
    return score_df

def map_accuracy(probs, truth, k=5, is_average=True):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        if is_average == True:
            # top accuracy
            correct = correct.float().sum(0, keepdim=False)
            correct = correct / len(truth)

            accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
            map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
            acc1 = accs[0]
            acc5 = accs[1]
            return map5, acc1, acc5

        else:
            return correct
