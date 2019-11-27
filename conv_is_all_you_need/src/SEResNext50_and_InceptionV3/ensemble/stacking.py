# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 14:45:56 2019

@author: Xuan-Laptop
"""

import pandas as pd
import numpy as np
from utils import np_macro_f1, encoding
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings
warnings.filterwarnings("ignore")

def calibration_flathead(y_val, p_pred):
    best_ts = 0
    best_f1 = 0
    for i in range(1, 51):
        ts = i/100
        out = np_macro_f1(y_val, (p_pred > ts).astype(int), return_details=False)
        if out > best_f1:
            best_f1 = out
            best_ts = ts
    df_res = np_macro_f1(y_val, (p_pred > best_ts).astype(int), return_details=True)
    return best_ts, df_res
    
def calibration_perclass(y_val, p_pred):
    ts_list = []
    for i in range(28):
        ts, _ = calibration_flathead(y_val[:, i], p_pred[:, i])
        ts_list.append(ts)
    df_res = np_macro_f1(y_val, (p_pred > ts_list).astype(int), return_details=True)
    return ts_list, df_res

stack_version = 'stacking_v4'

def getLogit(x, epsilon=1e-20):
    return np.log(x/(1 - x + epsilon) + epsilon)

def getProb(logit):
    return 1/(1 + np.log(-logit))

# load data and align
# chose one of the submission file, make sure id is the same as sample submission
submission = pd.read_csv('./data/sample_submission.csv')

# the files should be find in ./ref_data. It's the same as 5folds_v2
for i in range(5):
    if i == 0:
        df_val = pd.read_csv('./ref_data/fold_info_SEED1024_val_F{}.csv'.format(str(i)))
    else:
        df_val = df_val.append(pd.read_csv('./ref_data/fold_info_SEED1024_val_F{}.csv'.format(str(i))), ignore_index=True)
labels = df_val.Target.apply(encoding)
y_val = np.array(labels.tolist())


# put the names of models to be stacked here.
# two files should be included: model_name_val.csv and model_name_test.csv
# Model predicted probability for 28 classes of all images
# the val data will be aligned with respect to the val data loaded from ref_data
# the format would be: Id | 0 | 1 | ... | 28. 
models = [#'seresnext50', 
          'seresnext50_tta',
          'inceptionv3_tta',
          #'zhu',
          'zhu_614',
          #'zhu_Jan9',
          ]

p_test_all = []
p_val_all = []
res_details = []

mask = [str(x) for x in range(28)]
for i, model in enumerate(models):
    p_val = pd.read_csv('./data/{}_val.csv'.format(model))
    p_val = pd.merge(df_val[['Id']], p_val, how='left', on='Id')
    p_val_all.append(np.array(p_val[mask].values))
    df_res = np_macro_f1(y_val, np.array(p_val[mask].values), return_details=True)
    print('Model_%s f1 loss: %.4f'% (model, df_res.f1_scores.mean()))
    res_details.append(df_res)
    
    p_test = pd.read_csv('./data/{}_test.csv'.format(model))
    p_test_all.append(np.array(p_test[mask].values))

# Train 28 linear models for each class    
lr_models = []
coeff = []
for i in range(28):
    tmp = []
    for j in range(len(models)):
        tmp.append(p_val_all[j][:, i])
    X = np.array(tmp)
    Y = y_val[:, i:i+1]
    lr = LinearRegression()
    #lr = LogisticRegression()
    lr.fit(X.T, Y)
    lr_models.append(lr)
    coeff.append(lr.coef_[0])
coeff = np.array(coeff)

# Ensemble predictions
stacking_all = []
val_stack = []
for i in range(28):
    lr = lr_models[i]
    
    tmp = []
    for j in range(len(models)):
        tmp.append(p_test_all[j][:, i])
    X = np.array(tmp)
    Y = lr.predict(X.T)
    Y = Y.clip(0, 1)
    stacking_all.append(Y)
    
    tmp = []
    for j in range(len(models)):
        tmp.append(p_val_all[j][:, i])
    X_v = np.array(tmp)
    
    Y_v = lr.predict(X_v.T)
    Y_v = Y_v.clip(0, 1)
    val_stack.append(Y_v)
    
p_stack = np.squeeze(np.dstack(stacking_all))
p_stack_val = np.squeeze(np.dstack(val_stack))
df_stack = np_macro_f1(y_val, p_stack_val, return_details=True)
print('Stacking f1-loss: %4f' % (df_stack.f1_scores.mean()))

ts_flat, df_flat = calibration_flathead(y_val,  p_stack_val)
ts_perclass, df_perclass = calibration_perclass(y_val,  p_stack_val)

print('Flathead: %.4f, Per Class: %.4f' 
      %(np.mean(df_flat.f1_scores), np.mean(df_perclass.f1_scores)))

df_stack_val = df_val[['Id']]
df_stack_test = submission[['Id']]
for i in range(28):
    df_stack_val[str(i)] = p_stack_val[:, i]
    df_stack_test[str(i)] = p_stack[:, i]

df_coeff = pd.DataFrame(coeff)
df_coeff.columns = models

# store all necessary information
df_coeff.to_csv('{}_coef.csv'.format(stack_version), index=False)
df_stack_val.to_csv('{}_val.csv'.format(stack_version), index=False)
df_stack_test.to_csv('{}_test.csv'.format(stack_version), index=False)
    