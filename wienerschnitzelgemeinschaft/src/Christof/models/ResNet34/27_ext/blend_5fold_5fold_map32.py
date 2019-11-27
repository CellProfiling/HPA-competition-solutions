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

MODEL_PATH = 'Christof/models/ResNet34/27_ext/'

submit = pd.read_csv('Christof/assets/sample_submission.csv')
tta = 8

draw_predict = np.zeros((2,len(submit['Id']), 28))
draw_predict[0] = np.load(MODEL_PATH + 'pred_5fold.npy')
draw_predict[1] = np.load(MODEL_PATH + 'pred_5fold_32.npy')

draw_predict1 = (draw_predict[0] + draw_predict[1])/2
np.save(MODEL_PATH + f'blend_5fold_map32_5fold.npy',draw_predict1)
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
submit.to_csv(MODEL_PATH + 'blend_5fold_map32.csv', index=False)

from Christof.utils import f1_sub

best_sub = pd.read_csv('ens56d.csv')
f1_sub(best_sub,submit)
