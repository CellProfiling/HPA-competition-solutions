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

best_sub = pd.read_csv('best_sub.csv')
s0 = [s if isinstance(s, str) else '' for s in best_sub.Predicted]
p0 = [s.split() for s in s0]
y0 = np.zeros((best_sub.shape[0], 28)).astype(int)
for i in range(best_sub.shape[0]):
    for j in p0[i]: y0[i, int(j)] = 1

for i in range(28):
    desired[i] = y0[:,i].mean()


m1 = pd.read_csv('Christof/models/GAPNet/13_ext/oof_pred.csv')
m2 = pd.read_csv('Christof/models/ResNet34/27_ext/oof_pred.csv')
m3 = pd.read_csv('Christof/models/GAPNet/13_ext_60_40/oof_pred.csv')
m4 = pd.read_csv('Christof/models/GAPNet/14_ext/oof_pred.csv')
m5 = pd.read_csv('Christof/models/ResNet34/30/oof_pred.csv')

p1 = m1[m1.columns[3:]].values
p2 = m2[m2.columns[3:]].values
p3 = m3[m3.columns[3:]].values
p4 = m4[m4.columns[3:]].values
p5 = m5[m5.columns[3:]].values

#predicts = (p2 + 0.5*p4)
b1 = 0.5*p2 + 0.5*p3
b2 = 0.2*p1 + 0.25*p2 + .25*p3 + 0.3*p5

#predicts = p3
# custom thresholds to match lb proportions
def fit_thresh(predicts,desired):
    thresholds = np.linspace(0.95, 0.05, 101)
    pred = predicts.copy()
    for j in range(pred.shape[1]):
        for t in thresholds:
            pred[:, j] = (predicts[:, j] > t).astype(int)
            prop = np.mean(pred[:, j])
            if prop >= desired[j]: break
        #print(j, '%3.2f' % t, '%6.4f' % desired[j], '%6.4f' % prop, j, )
    return pred

ls = m1['Target'].values

def str2y(item):
    np_labels = np.zeros((28,))
    labels = item.split(' ')
    int_labels = [int(label) for label in labels]
    np_labels[int_labels] = 1
    return np_labels

ls2 = [str2y(item) for item in ls]
ls2 = np.array(ls2)

from sklearn.metrics import f1_score


for p in [p1,p2,p3,p4,p5]:
    print(f1_score(ls2,fit_thresh(p,desired), average='macro'))

print(f1_score(ls2,fit_thresh(b2,desired), average='macro'))

submit = pd.read_csv('Christof/assets/sample_submission.csv')
t1  = np.load('Christof/models/GAPNet/13_ext_60_40/pred_5fold_2.npy')
t2 = np.load('Christof/models/ResNet34/27_ext/pred_5fold_2.npy')
t3 = np.load('Christof/models/GAPNet/13_ext/pred_5fold_3.npy')
t4 = np.load('Christof/models/ResNet34/30/pred_5fold_2.npy')

draw_predict1 = 0.25*t1 + 0.25*t2 +0.25*t3 + 0.25*t4
#draw_predict1 = t2
MODEL_PATH = 'Christof/models/blends/2/'
np.save(MODEL_PATH + 'blend.npy',draw_predict1)
# custom thresholds to match lb proportions
thresholds = np.linspace(0.95, 0.05, 101)
pred = draw_predict1.copy()
for j in range(pred.shape[1]):
    for t in thresholds:
        pred[:, j] = (draw_predict1[:, j] > t).astype(int)
        prop = np.mean(pred[:, j])
        if prop >= desired[j]: break
    print(j, '%3.2f' % t, '%6.4f' % desired[j], '%6.4f' % prop, j, )

#print(pred[:5].astype(int))

label_predict = [np.arange(28)[score_predict == 1] for score_predict in pred]
str_predict_label = [' '.join(str(l) for l in lp) for lp in label_predict]

submit['Predicted'] = str_predict_label
# np.save('draw_predict_InceptionV3.npy', score_predict)
submit.to_csv(MODEL_PATH + 'submission_blend1.csv', index=False)

from Christof.utils import f1_sub

best_sub = pd.read_csv('ens56d.csv')
f1_sub(best_sub,submit)

s2 = pd.read_csv('Christof/models/blends/1/submission_blend1.csv')
f1_sub(best_sub,s2)