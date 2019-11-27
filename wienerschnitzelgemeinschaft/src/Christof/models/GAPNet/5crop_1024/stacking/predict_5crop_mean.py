import os, sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"



import numpy as np
import pandas as pd
from tqdm import tqdm


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

MODEL_PATH = 'Christof/models/GAPNet/5crop_1024/'

submit = pd.read_csv('Christof/assets/sample_submission.csv')
tta = 8

from Christof.utils import f1_sub

best_sub = pd.read_csv('ens56d.csv')
draw_predict = np.zeros((5,len(submit['Id']), 28))
for id, prefix in enumerate(['ul','ur','mm','bl','br']):
    draw_predict[id] = np.load(MODEL_PATH + f'pred1_{prefix}.npy')

    thresholds = np.linspace(0.95, 0.05, 101)
    pred = draw_predict[id].copy()
    for j in range(pred.shape[1]):
        for t in thresholds:
            pred[:, j] = (draw_predict[id][:, j] > t).astype(int)
            prop = np.mean(pred[:, j])
            if prop >= desired[j]: break

    label_predict = [np.arange(28)[score_predict == 1] for score_predict in pred]
    str_predict_label = [' '.join(str(l) for l in lp) for lp in label_predict]

    submit['Predicted'] = str_predict_label
    print(f1_sub(best_sub, submit))



# np.save('draw_predict_InceptionV3.npy', score_predict)
submit.to_csv(MODEL_PATH + 'submission_5crop_mean.csv', index=False)



#draw_predict1 = np.mean(draw_predict,axis = 0)
draw_predict1 = draw_predict[0] + draw_predict[1] +1.5*draw_predict[2] +draw_predict[3]+draw_predict[4]
draw_predict1 = draw_predict1 /5.5
np.save(MODEL_PATH + f'5crop_mean.npy',draw_predict1)
# custom thresholds to match lb proportions
thresholds = np.linspace(0.95, 0.05, 101)
pred = draw_predict1.copy()
for j in range(pred.shape[1]):
    for t in thresholds:
        pred[:, j] = (draw_predict1[:, j] > t).astype(int)
        prop = np.mean(pred[:, j])
        if prop >= desired[j]: break
    print(j, '%3.2f' % t, '%6.4f' % desired[j], '%6.4f' % prop, j, )

print(pred[:5].astype(int))

label_predict = [np.arange(28)[score_predict == 1] for score_predict in pred]
str_predict_label = [' '.join(str(l) for l in lp) for lp in label_predict]

submit['Predicted'] = str_predict_label
# np.save('draw_predict_InceptionV3.npy', score_predict)
submit.to_csv(MODEL_PATH + 'submission_5crop_mean.csv', index=False)



best_sub = pd.read_csv('ens56d.csv')
f1_sub(best_sub,submit)

