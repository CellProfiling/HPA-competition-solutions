import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import pandas as pd
import numpy as np

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

MODEL_PATH = 'Christof/models/GAPNet/13_ext_512crop/'

pred_ul = np.load(MODEL_PATH + 'pred_ul_40.npy')
pred_ur = np.load(MODEL_PATH + 'pred_ur_40.npy')
pred_mm = np.load(MODEL_PATH + 'pred_mm_40.npy')
pred_bl = np.load(MODEL_PATH + 'pred_bl_40.npy')
pred_br = np.load(MODEL_PATH + 'pred_br_40.npy')

X = np.stack([pred_ul,pred_ur,pred_mm,pred_bl,pred_br])
X = np.mean(X,axis=1)
X = np.transpose(X, (1, 0, 2))

from keras.models import load_model

preds = np.zeros((X.shape[0],28))
for f_id in range(5):
    m = load_model(MODEL_PATH + f'stacker{f_id}_40_stats.hdf5')
    preds += m.predict(X, batch_size = 512)

preds = preds/ 5

np.save(MODEL_PATH + 'pred_5fold_5crop_stacked.npy',preds)

#desired = {}
#best_sub = pd.read_csv('best_sub.csv')
#s0 = [s if isinstance(s, str) else '' for s in best_sub.Predicted]
#p0 = [s.split() for s in s0]
#y0 = np.zeros((best_sub.shape[0], 28)).astype(int)
#for i in range(best_sub.shape[0]):
#    for j in p0[i]: y0[i, int(j)] = 1

#for i in range(28):
#    desired[i] = y0[:,i].mean()

thresholds = np.linspace(0.95, 0.05, 101)
pred = preds.copy()
for j in range(pred.shape[1]):
    for t in thresholds:
        pred[:, j] = (preds[:, j] > t).astype(int)
        prop = np.mean(pred[:, j])
        if prop >= desired[j]: break
    print(j, '%3.2f' % t, '%6.4f' % desired[j], '%6.4f' % prop, j, )


print(pred[:5].astype(int))

label_predict = [np.arange(28)[score_predict == 1] for score_predict in pred]
str_predict_label = [' '.join(str(l) for l in lp) for lp in label_predict]

submit = pd.read_csv('Christof/assets/sample_submission.csv')
submit['Predicted'] = str_predict_label
# np.save('draw_predict_InceptionV3.npy', score_predict)
submit.to_csv(MODEL_PATH + 'submission_5crop_stacked_lb_prob_freq.csv', index=False)

from Christof.utils import f1_sub

best_sub = pd.read_csv('ens56d.csv')
f1_sub(best_sub,submit)

sub_650 = pd.read_csv('enstw43_10en36-642_10en39b-642_10hill2-636_10-646_2.4.csv')
f1_sub(sub_650,submit)

gap13_ext = pd.read_csv('Christof/models/GAPNet/13_ext/submission_loss_5fold_mean.csv')
gap13_ext_60_40 = pd.read_csv('Christof/models/GAPNet/13_ext_60_40/submission_loss_5fold_mean.csv')
f1_sub(gap13_ext_60_40,gap13_ext)

f1_sub(gap13_ext,submit)




preds2 = np.load('Christof/models/GAPNet/13_ext/pred_5fold.npy')

thresholds = np.linspace(0.95, 0.05, 101)
pred = preds2.copy()
for j in range(pred.shape[1]):
    for t in thresholds:
        pred[:, j] = (preds2[:, j] > t).astype(int)
        prop = np.mean(pred[:, j])
        if prop >= desired[j]: break
    print(j, '%3.2f' % t, '%6.4f' % desired[j], '%6.4f' % prop, j, )


print(pred[:5].astype(int))

label_predict = [np.arange(28)[score_predict == 1] for score_predict in pred]
str_predict_label = [' '.join(str(l) for l in lp) for lp in label_predict]

submit2 = pd.read_csv('Christof/assets/sample_submission.csv')
submit2['Predicted'] = str_predict_label

preds3 = 0.5 *preds + 0.5 * preds2


thresholds = np.linspace(0.95, 0.05, 101)
pred = preds3.copy()
for j in range(pred.shape[1]):
    for t in thresholds:
        pred[:, j] = (preds3[:, j] > t).astype(int)
        prop = np.mean(pred[:, j])
        if prop >= desired[j]: break
    print(j, '%3.2f' % t, '%6.4f' % desired[j], '%6.4f' % prop, j, )


print(pred[:5].astype(int))

label_predict = [np.arange(28)[score_predict == 1] for score_predict in pred]
str_predict_label = [' '.join(str(l) for l in lp) for lp in label_predict]

submit3 = pd.read_csv('Christof/assets/sample_submission.csv')
submit3['Predicted'] = str_predict_label
submit3.to_csv(MODEL_PATH + 'GAPNET blend_13_ext_13_ext5crop_stacked_lb_prob_freq.csv', index=False)



f1_sub(sub_650,submit)
f1_sub(sub_650,submit2)
f1_sub(sub_650,gap13_ext)
f1_sub(sub_650,submit3)

