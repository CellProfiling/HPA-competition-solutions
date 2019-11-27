import os, sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

best_sub = pd.read_csv('best_sub.csv')
s0 = [s if isinstance(s, str) else '' for s in best_sub.Predicted]
p0 = [s.split() for s in s0]
y0 = np.zeros((best_sub.shape[0], 28)).astype(int)
for i in range(best_sub.shape[0]):
    for j in p0[i]: y0[i, int(j)] = 1

for i in range(28):
    desired[i] = y0[:,i].mean()

MODEL_PATH = 'Christof/models/GAPNet/13_ext_512crop/'
oof_df = pd.read_csv(MODEL_PATH + 'oof_pred_ul.csv')

oof1 = pd.read_csv(MODEL_PATH + 'oof_pred_ul_40_40.csv')
oof2 = pd.read_csv(MODEL_PATH + 'oof_pred_ur_40_40.csv')
oof3 = pd.read_csv(MODEL_PATH + 'oof_pred_mm_40_40.csv')
oof4 = pd.read_csv(MODEL_PATH + 'oof_pred_bl_40_40.csv')
oof5 = pd.read_csv(MODEL_PATH + 'oof_pred_br_40_40.csv')

oof1 = oof1[oof1.columns[3:]].values
oof2 = oof2[oof2.columns[3:]].values
oof3 = oof3[oof3.columns[3:]].values
oof4 = oof4[oof4.columns[3:]].values
oof5 = oof5[oof5.columns[3:]].values

draw_predict1 = np.mean([oof1,oof2,oof3,oof4,oof5],axis = 0)
np.save(MODEL_PATH + f'oof.npy',draw_predict1)
# custom thresholds to match lb proportions
thresholds = np.linspace(0.95, 0.05, 101)
pred = draw_predict1.copy()
for j in range(pred.shape[1]):
    for t in thresholds:
        pred[:, j] = (draw_predict1[:, j] > t).astype(int)
        prop = np.mean(pred[:, j])
        if prop >= desired[j]: break
    print(j, '%3.2f' % t, '%6.4f' % desired[j], '%6.4f' % prop, j, )

ls = oof_df['Target'].values

def str2y(item):
    np_labels = np.zeros((28,))
    labels = item.split(' ')
    int_labels = [int(label) for label in labels]
    np_labels[int_labels] = 1
    return np_labels

ls2 = [str2y(item) for item in ls]
ls2 = np.array(ls2)

from sklearn.metrics import f1_score

f1_score(ls2,pred, average='macro')


from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Lambda, Concatenate
import keras.backend as K
from keras.models import Model

def build_stacker():
    inp = Input((5,28,))
    x = Flatten()(inp)
    x2 = Lambda(lambda x: K.mean(x,axis=1))(inp)
    x3 = Lambda(lambda x: K.std(x, axis=1))(inp)
    x4  = Lambda(lambda x: K.max(x, axis=1))(inp)
    x = Concatenate()([x,x2,x3,x4])
    #x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    #x = Dense(64, activation='relu')(x)
    #x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(28, activation='sigmoid')(x)

    m = Model(inp,out)

    m.compile(optimizer='adam',loss='binary_crossentropy')
    return m

X = np.stack([oof1,oof2,oof3,oof4,oof5])
X = np.transpose(X, (1, 0, 2))
y = ls2

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, random_state=23)

s_id = -1
p2 = np.zeros((X.shape[0],28))
for tr_ind, val_ind in kf.split(X,y,):
    s_id +=1
    X_train = X[tr_ind]
    y_train = y[tr_ind]
    X_val = X[val_ind]
    y_val = y[val_ind]
    m = build_stacker()
    m.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=20, batch_size=128)
    m.save(MODEL_PATH + f'stacker{s_id}_40_stats.hdf5')
    p2[val_ind] = m.predict(X_val)

pred_val = np.mean(X, axis = 1)
thresholds = np.linspace(0.95, 0.05, 101)
pred = pred_val.copy()
for j in range(pred.shape[1]):
    for t in thresholds:
        pred[:, j] = (pred_val[:, j] > t).astype(int)
        prop = np.mean(pred[:, j])
        if prop >= desired[j]: break
    print(j, '%3.2f' % t, '%6.4f' % desired[j], '%6.4f' % prop, j, )

f1_score(y,pred,average='macro')

#p2 = m.predict(X_val)
thresholds = np.linspace(0.95, 0.05, 101)
pred = p2.copy()
for j in range(pred.shape[1]):
    for t in thresholds:
        pred[:, j] = (p2[:, j] > t).astype(int)
        prop = np.mean(pred[:, j])
        if prop >= desired[j]: break
    print(j, '%3.2f' % t, '%6.4f' % desired[j], '%6.4f' % prop, j, )

f1_score(y,pred,average='macro')



