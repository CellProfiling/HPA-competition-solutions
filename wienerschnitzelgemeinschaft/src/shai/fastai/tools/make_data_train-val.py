from fastai.conv_learner import *
from fastai.dataset import *
from tensorboard_cb_old import *
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scipy.optimize as opt
from sklearn.model_selection import StratifiedKFold
from itertools import chain
from collections import Counter
import pickle as pkl

import warnings
warnings.filterwarnings("ignore")

#=======================================================================================================================
# Something
#=======================================================================================================================

PATH = './'
TRAIN = '../input/train/'
TEST = '../input/test/'
LABELS = '../input/train.csv'
LABELS_ext = '../input/HPAv18RBGY_wodpl.csv'
SAMPLE = '../input/sample_submission.csv'

name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',
14:  'Microtubules',
15:  'Microtubule ends',
16:  'Cytokinetic bridge',
17:  'Mitotic spindle',
18:  'Microtubule organizing center',
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',
22:  'Cell junctions',
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',
27:  'Rods & rings' }

#=======================================================================================================================
# Data
#=======================================================================================================================
# image_df = pd.read_csv(LABELS)
# image_df = image_df[(image_df.Id != 'dc756dea-bbb4-11e8-b2ba-ac1f6b6435d0') &
#                     (image_df.Id != 'c861eb54-bb9f-11e8-b2b9-ac1f6b6435d0') &
#                     (image_df.Id != '7a88f200-bbc3-11e8-b2bc-ac1f6b6435d0')]


image_df = pd.read_csv(LABELS_ext)
image_df = image_df[(image_df.Id != '27751_219_G10_1') ]

image_df['target_list'] = image_df['Target'].map(lambda x: [int(a) for a in x.split(' ')])

all_labels = list(chain.from_iterable(image_df['target_list'].values))
c_val = Counter(all_labels)
n_keys = c_val.keys()
max_idx = max(n_keys)

#==================================================================================
# visualize train distribution
fig, ax1 = plt.subplots(1,1, figsize = (10, 5))
ax1.bar(n_keys, [c_val[k] for k in n_keys])
ax1.set_xticks(range(max_idx))
ax1.set_xticklabels([name_label_dict[k] for k in range(max_idx)], rotation=90)
plt.show()
#==================================================================================
for k,v in c_val.items():
    print(name_label_dict[k], 'count:', v)

# create a categorical vector
image_df['target_vec'] = image_df['target_list'].map(lambda ck: [i in ck for i in range(max_idx+1)])


raw_train_df, valid_df = train_test_split(image_df,
                 test_size = 0.15,
                  # hack to make stratification work
                 stratify = image_df['Target'].map(lambda x: x[:3] if '27' not in x else '0'),
                                          random_state= 42)

print(raw_train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')

# #=================================================================================
# # # Balance data
# #================================================================================

TRAIN_IMAGES_PER_CATEGORY=50
out_df_list = []
for k,v in c_val.items():
    if v>40:
        keep_rows = raw_train_df['target_list'].map(lambda x: k in x)
        out_df_list += [raw_train_df[keep_rows].sample(TRAIN_IMAGES_PER_CATEGORY,
                                                       replace=True)]
train_df = pd.concat(out_df_list, ignore_index=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
train_sum_vec = np.sum(np.stack(train_df['target_vec'].values, 0), 0)
valid_sum_vec = np.sum(np.stack(valid_df['target_vec'].values, 0), 0)
ax1.bar(n_keys, [train_sum_vec[k] for k in n_keys])
ax1.set_title('Training Distribution')
ax2.bar(n_keys, [valid_sum_vec[k] for k in n_keys])
ax2.set_title('Validation Distribution')
plt.show()
#=======================================================================================================================
train_df.to_csv('../input/train_ext_balanced.csv', index=False)
raw_train_df.to_csv('../input/train_ext_unbalanced.csv', index=False)
valid_df.to_csv('../input/valid_ext_unbalanced.csv', index=False)

tr_n = raw_train_df['Id'].values.tolist()
val_n = valid_df['Id'].values.tolist()
tr_n = tr_n[:-2]  # pytorch has problems if last batch has one sample

test_names = list({f[:36] for f in os.listdir(TEST)})