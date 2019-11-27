import os, sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"



import numpy as np
import pandas as pd
from tqdm import tqdm


import numpy as np
import pandas as pd

eps = 0.004
desired = {
    0: 0.36239782,
    1: 0.043841336,
    2: 0.075268817,
    3: 0.059322034,
    4: 0.075268817,
    5: 0.075268817,
    6: 0.043841336,
    7: 0.075268817,
    8: eps,
    9: eps,
    10: eps,
    11: 0.043841336,
    12: 0.043841336,
    13: 0.014198783,
    14: 0.043841336,
    15: eps,
    16: 0.028806584,
    17: 0.014198783,
    18: 0.028806584,
    19: 0.059322034,
    20: eps,
    21: 0.126126126,
    22: 0.028806584,
    23: 0.075268817,
    24: eps,
    25: 0.222493888,
    26: 0.028806584,
    27: eps
}

MODEL_PATH = 'Christof/models/GAPNet/13_ext_60_40/'
oof_pred = pd.read_csv(MODEL_PATH + 'oof_pred.csv')
s0 = [s if isinstance(s, str) else '' for s in oof_pred.Target]
p0 = [s.split() for s in s0]
y0 = np.zeros((oof_pred.shape[0], 28)).astype(int)
for i in range(oof_pred.shape[0]):
    for j in p0[i]: y0[i, int(j)] = 1

from sklearn.metrics import roc_curve, precision_recall_curve
def threshold_search(y_true, y_proba, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001)
    F = 2 / (1/precision + 1/recall)
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    search_result = {'threshold': best_th , 'f1': best_score}
    return search_result

threshs = np.zeros((28,))
for i in range(28):
    y_proba = oof_pred[f'pred_{i}'].values
    y_true = y0[:,i]
    s = threshold_search(y_true, y_proba, plot=False)
    threshs[i] = s['threshold']

preds = oof_pred[oof_pred.columns[3:]].values
for i in range(28):
    preds[:,i] = preds[:,i] > threshs[i]
from sklearn.metrics import f1_score
f1_score(y0,preds, average='macro')

submit = pd.read_csv('Christof/assets/sample_submission.csv')
tta = 8

draw_predict = np.zeros((5,len(submit['Id']), 28))
for fold_id in range(5):
    draw_predict[fold_id] = np.load(MODEL_PATH + f'pred{fold_id}.npy')


draw_predict1 = np.mean(draw_predict,axis = 0)
np.save(MODEL_PATH + f'pred_5fold_3.npy',draw_predict1)
pred = draw_predict1.copy()
for i in range(28):
    pred[:,i] =(pred[:,i] > threshs[i]).astype(int)


label_predict = [np.arange(28)[score_predict == 1] for score_predict in pred]
str_predict_label = [' '.join(str(l) for l in lp) for lp in label_predict]

submit['Predicted'] = str_predict_label
# np.save('draw_predict_InceptionV3.npy', score_predict)
submit.to_csv(MODEL_PATH + 'submission_loss_5fold_oof_treshs.csv', index=False)

from Christof.utils import f1_sub

best_sub = pd.read_csv('ens56d.csv')
f1_sub(best_sub,submit)

old_sub = pd.read_csv(MODEL_PATH + 'submission_loss_5fold_mean.csv')
f1_sub(best_sub,old_sub)