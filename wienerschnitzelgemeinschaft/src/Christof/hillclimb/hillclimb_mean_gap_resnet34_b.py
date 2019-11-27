#import matplotlib as mpl
#mpl.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.special import expit
import os

PATH_TO_TARGET = 'Christof/assets/train.csv'
PATH_TO_SUB = 'Christof/assets/sample_submission.csv'
nfold = 5

LABEL_MAP = {
0: "Nucleoplasm" ,
1: "Nuclear membrane"   ,
2: "Nucleoli"   ,
3: "Nucleoli fibrillar center",
4: "Nuclear speckles"   ,
5: "Nuclear bodies"   ,
6: "Endoplasmic reticulum"   ,
7: "Golgi apparatus"  ,
8: "Peroxisomes"   ,
9:  "Endosomes"   ,
10: "Lysosomes"   ,
11: "Intermediate filaments"  ,
12: "Actin filaments"   ,
13: "Focal adhesion sites"  ,
14: "Microtubules"   ,
15: "Microtubule ends"   ,
16: "Cytokinetic bridge"   ,
17: "Mitotic spindle"  ,
18: "Microtubule organizing center",
19: "Centrosome",
20: "Lipid droplets"   ,
21: "Plasma membrane"  ,
22: "Cell junctions"   ,
23: "Mitochondria"   ,
24: "Aggresome"   ,
25: "Cytosol" ,
26: "Cytoplasmic bodies",
27: "Rods & rings"}

df = pd.read_csv(PATH_TO_TARGET)
df.set_index('Id',inplace=True)
print(df.head())
print(df.shape)

file_list = list(df.index.values)

ss = pd.read_csv(PATH_TO_SUB)
ss.set_index('Id',inplace=True)
print(ss.head())
print(ss.shape)

oof = df.copy()
russ_mnames = ['preresnet_s67'] #'airx_s67'
for k,m in enumerate(russ_mnames):
    first = True
    for fold in range(nfold):
        idf, pooff, yooff = pickle.load(open('Christof/hillclimb/oof/russ/' + m + '_' + str(fold) + '.pkl', 'rb'))
        # change to probability scale
        pooff = expit(pooff)

        if first:
            id = idf
            poof = pooff.copy()
            yoof = yooff.copy()
            first = False
        else:
            id = id + idf
            poof = np.concatenate((poof, pooff))
            yoof = np.concatenate((yoof, yooff))

    mdf = pd.DataFrame({'Id': id})
    # print(poof.shape)

    if k==0:mdf['y'] = [yoof[i] for i in range(yoof.shape[0])]
    mdf[m] = [poof[i,:,0,0] for i in range(poof.shape[0])]
    mdf.set_index('Id', inplace=True)
    oof = oof.join(mdf)

oof.head()

# add GAPNET_13_512crop

gap512oof = np.load('Christof/hillclimb/oof/GAPNet_13_512crop/oof_stacked.npy')
print(gap512oof.shape)

oof['GAPNet_13_512crop'] = [gap512oof[i] for i in range(gap512oof.shape[0])]

cmnames = ['GAPNet_13_ext','GAPNet_13_ext_rgby','GAPNet_13_ext6040','ResNet34_27_ext','ResNet34_30']


for cm in cmnames:
    cmoof = pd.read_csv('Christof/hillclimb/oof/' + cm + '/oof_pred.csv')

    cmof = pd.DataFrame({'Id': cmoof['Id'].values})
    vals = cmoof[cmoof.columns[3:]].values
    cmof[cm] = [vals[i] for i in range(vals.shape[0])]
    cmof.set_index('Id', inplace=True)

    oof = oof.join(cmof)

vp = oof.loc[oof['y'].notnull()]
offset = 2

vp2 = vp.copy()

vp2.head()
y = np.array([a for a in vp2.y])

tv1 = np.array(list(vp['GAPNet_13_ext'].values))
tv2 = np.array(list(vp['GAPNet_13_ext6040'].values))
tv3 = np.array(list(vp['GAPNet_13_ext_rgby'].values))
tv = np.mean([tv1,tv2,tv3],axis = 0)
vp2['GAPNET1'] = [tv[i] for i in range(tv.shape[0])]
vp2.drop(['GAPNet_13_ext','GAPNet_13_ext6040','GAPNet_13_ext_rgby'],inplace = True, axis = 1)

tv1 = np.array(list(vp['ResNet34_27_ext'].values))
tv2 = np.array(list(vp['ResNet34_30'].values))
tv = np.mean([tv1,tv2],axis = 0)
vp2['ResNet34'] = [tv[i] for i in range(tv.shape[0])]
vp2.drop(['ResNet34_27_ext','ResNet34_30'],inplace = True, axis = 1)

tv1 = np.array(list(vp2['GAPNET1'].values))
tv2 = np.array(list(vp2['ResNet34'].values))
tv = np.mean([tv1,tv2],axis = 0)
vp2['GapNet34'] = [tv[i] for i in range(tv.shape[0])]
vp2.drop(['GAPNET1','ResNet34'],inplace = True, axis = 1)

mnames = [m for m in vp2.columns[2:].values]
nm = len(mnames)

np.set_printoptions(precision=3, linewidth=100, suppress=True)
f = []
for m in mnames:
    f.append(np.array(list(vp2[m])).flatten())
print(mnames)
r = np.corrcoef(f)
print(r)

from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
def threshold_search(y_true, y_proba, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001)
    F = 2 / (1/precision + 1/recall)
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    search_result = {'threshold': best_th , 'f1': best_score}
    return search_result


cmthresh = {}
for j in range(nm):
    mname = vp2.columns[j + offset]
    tv = np.array(list(vp2.iloc[:, offset + j].values))
    threshs = np.zeros((28,))
    f1_scores = np.zeros((28,))
    for i in range(28):
        y_proba = tv[:,i]
        y_true = y[:,i]
        s = threshold_search(y_true, y_proba, plot=False)
        threshs[i] = s['threshold']
        f1_scores[i] = s['f1']
    print(mname,np.mean(f1_scores))
    cmthresh[mname] = threshs


# oof = small

# hillclimbing
# vp = oof.copy()
# offset = 5
# nm = vp.shape[1] - offset
nr = vp2.shape[0]
y = np.array([y for y in vp2.y])
# y = np.array(list(vp['masks'].values))
nstep = 50

step0fix = -1
# step0fix = 5
step1fix = -1

intercept = 0
rez = pd.DataFrame({'step': np.zeros(nstep).astype(int), 'midx': np.zeros(nstep).astype(int),
                    'score': np.zeros(nstep)})
rez.columns = ['step','midx','score']
currEnsemble = 0 * np.array(list(vp2.iloc[:, offset].values))

metric = 'f1'
# metric = 'iou'

# using model-specific thresholds below so set global to 0
# threshold = 0.
# threshold = -0.4
threshold = 0

print('using ' + metric + ' threshold', threshold)

# maximizing
bestScore = -9999999
single = pd.DataFrame({'mod': np.repeat("", nm)})
start = 0
for step in range(start, nstep):
    single[metric + '_' + str(step)] = np.nan

currIndex = 0
currScore = -9999999
for step in range(start, nstep):


    for i in range(nm):

        mname = vp2.columns[i + offset]

        tv = np.array(list(vp2.iloc[:, offset + i].values))

        # adjust by model-specific threshold
        #tv = tv - mthresh[mname]

        # adjust by model-class-specific thresholds
        for j in range(tv.shape[1]):
            tv[:,j] = tv[:,j] - cmthresh[mname][j]

        #if np.sum(np.isnan(tv)) > 0: continue

        if step == 0:
            tryEnsemble = tv
        else:
            # running mean
            tryEnsemble = (step * currEnsemble + tv) / (step + 1)
            # full matrix
            # choose median or mean here
            # tryEnsemble = rowMedians(cbind(currEnsemble,tv))
            # tryEnsemble = rowMeans(cbind(currEnsemble,tv))
            # quantiles are a lot slower
            # tryEnsemble = rowQuantiles(cbind(currEnsemble,tv),probs=0.6)

        if metric == "f1":
            tryScore = f1_score(y, np.int32(tryEnsemble > threshold),
                                average='macro')
            if step > 0: print('    try', i, mname, "%6.4f" % tryScore)

        ok = 1
        if (step == 0) and (step0fix > -1) and (i != step0fix): ok = 0
        if (step == 1) and (step1fix > -1) and (i != step1fix): ok = 0

        if (tryScore > currScore) and (ok == 1):
            currIndex = i
            currScore = tryScore
            if (step == 0):
                saveEnsemble = tv.copy()
            else:
                # running mean
                saveEnsemble = tryEnsemble.copy()
                # save whole matrix
                # saveEnsemble = np.concatenate((currEnsemble,tv),axis=1)

        # if (step==0) or ((step==1) and (step0fix>-1)) or ((step==2) and (step1fix>-1)):
        if (step == 0):
            print("model", i, metric, "for", vp2.columns[i + offset], "= %6.4f" % tryScore)

        if step == 0:
            single.loc[i, 'mod'] = vp2.columns[i + offset]
        single.iloc[i, step + 1] = tryScore

    print("step =", step, "  index =", currIndex, "  name =", vp2.columns[currIndex + offset], "  ",
              metric, "= %6.4f" % currScore)

    currEnsemble = saveEnsemble.copy()
    if currScore > bestScore:
        bestScore = currScore
        bestStep = step
        bestEnsemble = currEnsemble.copy()

    rez.iloc[step, 0] = step
    rez.iloc[step, 1] = currIndex
    rez.iloc[step, 2] = currScore

rez0 = rez.copy()

# tb = table(rez$index[1:bestStep])
# names(tb) = names(vp)[(offset+1):ncol(vp)][as.integer(names(table(rez$index[1:bestStep])))]

# if (intercept!=0) {
#   tb["Intercept"] = intercept
# }

# print(tb[order(-tb)])
# print(single[order(-single[,2]),][1:min(nrow(single),40),])

# plot hillclimbing
print('')
plt.plot(rez0.step, rez0.score)
plt.plot(bestStep, bestScore, "xr", label="Best")
plt.xlabel("Step")
plt.ylabel("f1")
plt.title("f1 Hillclimbing ({}, {})".format(bestStep, bestScore))
plt.legend()
plt.show()
#plt.gcf().clear()

# display the weights
rez = rez.iloc[:(bestStep + 1)]
# print(rez)

single0 = single.iloc[:, :2]

rez['midx'] = rez['midx'].astype(int)
rez = rez.join(single0, how='left', on='midx')
print(rez)
print('')

rez['weight'] = 1
tb = rez.loc[:, ['midx', 'weight']].groupby('midx').count()
# print(tb)
# print(single)

tb = tb.join(single0, how='left')
print(tb)