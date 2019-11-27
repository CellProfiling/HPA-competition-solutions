# coding: utf-8
import pandas as pd
import numpy as np
import os

res18_256_test = np.load('../model_weights/res18_256/test/res18_256-epoch_142_loss_0.4630_cv_0.6798_model.pth-prob-tencrop.npy')
res18_384_test = np.load('../model_weights/res18_384/test/res18_384-epoch_147_loss_0.4510_cv_0.6880_model.pth-prob-tencrop.npy')
res18_512_test = np.load('../model_weights/res18_512/test/res18_512-epoch_106_loss_0.5043_cv_0.6382_model.pth-prob-tencrop.npy')

sample_sub_df = pd.read_csv('../input/sample_submission.csv', engine='python')

thredshold = np.ones(28) * 0.2
all_pred = []
all_id = sample_sub_df.Id.values
all_prob = res18_256_test*0.5 + res18_384_test*0.25 + res18_512_test*0.25
for prob in all_prob:
    s = ' '.join(list([str(i) for i in np.nonzero(prob>thredshold)[0]]))
    if s == '':
        s = str(prob.argmax())
    all_pred.append(s)
sub_df = pd.DataFrame({ 'Id' : all_id , 'Predicted' : all_pred}).astype(str)
sub_df.to_csv('../sub/weighted_ensemble_256_384_512.csv', header=True, index=False)


thredshold = np.ones(28) * 0.2
all_pred = []
all_id = sample_sub_df.Id.values
all_prob = (res18_256_test + res18_384_test + res18_512_test)/3.0
for prob in all_prob:
    s = ' '.join(list([str(i) for i in np.nonzero(prob>thredshold)[0]]))
    if s == '':
        s = str(prob.argmax())
    all_pred.append(s)
sub_df = pd.DataFrame({ 'Id' : all_id , 'Predicted' : all_pred}).astype(str)
sub_df.to_csv('../sub/avg_ensemble_256_384_512.csv', header=True, index=False)

