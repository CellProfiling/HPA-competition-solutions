#============ Basic imports ============#e
import os
import time
import pandas as pd
import gc
import cv2
import csv
import random
import json
from sklearn.metrics.ranking import roc_auc_score
import numpy as np
from sklearn.metrics import log_loss
import sys
sys.path.insert(0, '..')  
from sklearn.metrics import f1_score

# compute f1 score
def f1_score_mine(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    return f1_score(y_true, y_pred)

# search the threshold which was used to obtain the maximum f1 score for each class
def search_f1(output, target):
    max_result_f1_list = []
    max_threshold_list = []
    for i in range(output.shape[1]):
        output_class = output[:, i]
        target_class = target[:, i]
        max_result_f1 = 0
        max_threshold = 0
        for threshold in [x * 0.01 for x in range(0, 100)]:
            prob = output_class > threshold
            label = target_class
            result_f1 = f1_score_mine(label, prob)
            if result_f1 > max_result_f1:
                max_result_f1 = result_f1
                max_threshold = threshold
        max_result_f1_list.append(round(max_result_f1,3))
        max_threshold_list.append(max_threshold)

    return max_threshold_list, max_result_f1_list

def apply_threasholds(y_pred, threasholds):
    temp = y_pred.copy()
    for i, value in enumerate(threasholds):
        temp[:, i] = temp[:, i] > value
    return temp

if __name__ == '__main__':

    with open('SETTINGS.json', 'r') as f:
        path_dict = json.load(f)    

    # read precomputed neural network predictions
    ensemble_path = path_dict['TRAINED_MODEL_CHECKPOINT_DIR'] + 'ensemble/'
    KAGGLE_LABEL_TRAIN_DATA_CLEAN_PATH = path_dict['KAGGLE_LABEL_TRAIN_DATA_CLEAN_PATH']
    df_all = pd.read_csv(KAGGLE_LABEL_TRAIN_DATA_CLEAN_PATH)
    val_truth_oof = df_all.sort_values(by='Id').reset_index(drop = True).loc[:, 'Nucleoplasm':'Rods & rings'].values
    models_list = ['dense121_v2', 'dense121_v3', 'seresnextnet50', 'dense121_v5', 'xception_osmr',  'dense169', 'ibn_dense121']
    df_val_list = [pd.read_csv(ensemble_path + x + '_val.csv').loc[:, '0':'27'].values for x in models_list]
    df_test_list = [pd.read_csv(ensemble_path + x + '_test.csv').loc[:, '0':'27'].values for x in models_list]

    val_average = sum(df_val_list)/len(df_val_list)
    # search the threshold 
    max_threshold, max_result_f1 = search_f1(val_average, val_truth_oof)
    test_aug_oof = sum(df_test_list)/len(df_test_list)

    # produce submission file 
    df_test = pd.read_csv(path_dict['SUBMIT_CSV_PATH'])
    with open(path_dict['SUBMISSION_DIR'] + 'submit_KFOLD_aug_oof_0.7_7.csv', 'w') as file:
        file.write('Id,Predicted'+"\n") 
        for i in range(df_test.shape[0]):
            
            id_s = df_test['Id'][i]
            vote_s = test_aug_oof[i].tolist()
            vote_s_index = np.where(np.array(vote_s)>np.array(max_threshold)*0.7)[0]

            out_str = id_s + ','
            out_str = out_str + ' '.join(str(l) for l in vote_s_index)
            if vote_s_index.shape[0] == 0:
                vote_s_index = np.where(np.array(vote_s)>np.array(max_threshold)*0.6)[0]
                out_str = id_s + ','
                out_str = out_str + ' '.join(str(l) for l in vote_s_index)
                if vote_s_index.shape[0] == 0:
                    vote_s_index = np.where(np.array(vote_s)>np.array(max_threshold)*0.5)[0]
                    out_str = id_s + ','
                    out_str = out_str + ' '.join(str(l) for l in vote_s_index)
         
            file.write(out_str+"\n")

