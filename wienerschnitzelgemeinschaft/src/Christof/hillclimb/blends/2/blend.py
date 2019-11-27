import pandas as pd
import numpy as np
import pickle
from scipy.special import expit




# 0          8            airx_s67  0.703696
# 1          5       preresnet_s67  0.697541
# 2          5   GAPNet_13_512crop  0.685118
# 3          1       GAPNet_13_ext  0.669337
# 4          3  GAPNet_13_ext_rgby  0.650539
# 5          1   GAPNet_13_ext6040  0.683339
# 6          1     ResNet34_27_ext  0.664597
# 7          3         ResNet34_30  0.669874

MODEL_PATH = 'Christof/hillclimb/blends/2/'
nfold = 5
russ_mnames = ['airx_s67','preresnet_s67']

sub = pd.read_csv('Christof/assets/sample_submission.csv')
sub.set_index('Id',inplace=True)
# for k,m in enumerate(russ_mnames):
#     first = True
#     for fold in range(nfold):
#         idf, pooff, yooff = pickle.load(open('Christof/hillclimb/preds/russ/' + m + '_' + str(fold) + '_mm.pkl', 'rb'))
#         # change to probability scale
#         pooff = expit(pooff)
#
#         if first:
#             id = idf
#             poof = pooff.copy()
#             yoof = yooff.copy()
#             first = False
#         else:
#             #id = id + idf
#             poof += pooff
#             #yoof = np.concatenate((yoof, yooff))
#     poof = poof / nfold
#     mdf = pd.DataFrame({'Id': id})
#     # print(poof.shape)
#
#     #if k==0:mdf['y'] = [yoof[i] for i in range(yoof.shape[0])]
#     mdf[m] = [poof[i,:,0,0] for i in range(poof.shape[0])]
#     mdf.set_index('Id', inplace=True)
#
#     sub = sub.join(mdf)

models = ['GAPNet_13_512crop','GAPNet_13_ext','GAPNet_13_ext6040','ResNet34_27_ext','ResNet34_30']


model_paths = ['GAPNet/13_ext_512crop/','GAPNet/13_ext/','GAPNet/13_ext_60_40/']
model_paths += ['ResNet34/27_ext/','ResNet34/30/']

for i,m in enumerate(models):
    path = model_paths[i]
    p = np.load('Christof/models/' + path + 'pred_5fold.npy')
    sub[m] = [p[i, :] for i in range(p.shape[0])]

nm = 5
offset = 1

weights = np.array([8,2,7,3,4])

tv = np.zeros((p.shape))
for j in range(nm):
    tv += weights[j] * np.array(list(sub.iloc[:, offset + j].values))

tv = tv / np.sum(weights)


np.save(MODEL_PATH + 'hc.npy',tv)


desired = {}
best_sub = pd.read_csv('enstw43_10en36-642_10en39b-642_10hill2-636_10-646_2.4.csv')
s0 = [s if isinstance(s, str) else '' for s in best_sub.Predicted]
p0 = [s.split() for s in s0]
y0 = np.zeros((best_sub.shape[0], 28)).astype(int)
for i in range(best_sub.shape[0]):
    for j in p0[i]: y0[i, int(j)] = 1

for i in range(28):
    desired[i] = y0[:,i].mean()

# custom thresholds to match lb proportions
thresholds = np.linspace(0.95, 0.05, 101)
pred = tv.copy()
for j in range(pred.shape[1]):
    for t in thresholds:
        pred[:, j] = (tv[:, j] > t).astype(int)
        prop = np.mean(pred[:, j])
        if prop >= desired[j]: break
    print(j, '%3.2f' % t, '%6.4f' % desired[j], '%6.4f' % prop, j, )

print(pred[:5].astype(int))

label_predict = [np.arange(28)[score_predict == 1] for score_predict in pred]
str_predict_label = [' '.join(str(l) for l in lp) for lp in label_predict]

submit = pd.read_csv('Christof/assets/sample_submission.csv')
submit['Predicted'] = str_predict_label
# np.save('draw_predict_InceptionV3.npy', score_predict)
submit.to_csv(MODEL_PATH + 'submission_hc.csv', index=False)

from Christof.utils import f1_sub

good_sub = pd.read_csv('ens56d.csv')
print(f1_sub(good_sub,submit))

best_sub = pd.read_csv('enstw43_10en36-642_10en39b-642_10hill2-636_10-646_2.4.csv')
print(f1_sub(best_sub,submit))

sub2 = pd.read_csv('Christof/hillclimb/blends/2/submission_hc.csv')
print(f1_sub(best_sub,sub2))
