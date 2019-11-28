import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from skimage.transform import resize
from skimage.color import rgb2gray, gray2rgb

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from tqdm import tqdm_notebook

import gc
import math
import sys

from fastai import *
from fastai.vision import *

np.random.seed(42)

data_dir = '../../../input/'
submit_l1_dir = "../../../submits/"
weights_dir = "../../weights/"
results_dir = '../../../results/'

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

def kfold_threshold(y_true, y_pred):
    n_classes = len(name_label_dict)
    classes_thresholds = []
    classes_scores = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=239)
    for i in range(n_classes):
        kf_class_thresholds = []
        if (sum(y_true[:,i]) > 20):
            for _, tst_inx in kf.split(y_true,y_true[:,i]):
                t_min = np.min(y_pred[tst_inx,i])
                t_max = np.max(y_pred[tst_inx,i])
                thresholds = np.linspace(t_min, t_max, 50)
                scores = np.array([
                    f1_score(y_true[tst_inx,i], np.int32(y_pred[tst_inx,i] >= threshold)) for threshold in thresholds
                ])
                threshold_best_index = np.argmax(scores)
                kf_class_thresholds.append(thresholds[threshold_best_index])
            threshold = np.mean(kf_class_thresholds)
            classes_thresholds.append(threshold)
            f1 = f1_score(y_true[:,i], np.int32(y_pred[:,i] >= threshold))
            classes_scores.append(f1)
        else:
            t_min = np.min(y_pred[:,i])
            t_max = np.max(y_pred[:,i])
            thresholds = np.linspace(t_min, t_max, 50)
            scores = np.array([
                f1_score(y_true[:,i], np.int32(y_pred[:,i] >= threshold)) for threshold in thresholds
            ])
            threshold_best_index = np.argmax(scores)
            classes_thresholds.append(thresholds[threshold_best_index])
            f1 = f1_score(y_true[:,i], np.int32(y_pred[:,i] >= thresholds[threshold_best_index]))
            classes_scores.append(f1)
    return classes_thresholds, classes_scores