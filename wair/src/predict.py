#============ Basic imports ============#e
import os
import time
import pandas as pd
import gc
import cv2
import csv
import random
import json
import sys
from glob import glob
from sklearn.metrics.ranking import roc_auc_score
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
from sklearn.metrics import log_loss
import torch.utils.data as data

from src.models.models import *
from src.dataset.dataset import *
from src.tuils.tools import *
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import albumentations

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

# create dataset class        
class PredictionDatasetAug:
    def __init__(self, name_list, df_data, n_test_aug, mode):
        self.name_list = name_list
        self.df_data = df_data
        self.n_test_aug = n_test_aug
        self.mode = mode

    def __len__(self):
        return len(self.name_list) * self.n_test_aug

    def __getitem__(self, idx):
        name = self.name_list[idx % len(self.name_list)]
        if self.mode == 'val':
            image_path = path_dict['ALL_IMAGE_TRAIN_DATA_CLEAN_DIR'] + '{name}.png'
            label = torch.FloatTensor(self.df_data[self.df_data['Id']==name].loc[:, 'Nucleoplasm':'Rods & rings'].values)
        if self.mode == 'test':
            image_path = path_dict['IMAGE_TEST_DATA_CLEAN_DIR'] + '{name}.png'
            label = torch.FloatTensor(self.df_data[self.df_data['Id']==name].loc[:, 'Nucleoplasm':'Rods & rings'].values)
        image = cv2.imread(image_path.format(name=name), -1)
        image = image[:,:,0:3]
        image = valid_transform_aug(image=image)['image']

        image = image.transpose(2, 0, 1)
        
        return name, image, label

# inference
def predict(model, name_list, df_data, batch_size: int, n_test_aug: int, aug=False, mode='val'):
    if aug:
        loader = DataLoader(
            dataset=PredictionDatasetAug(name_list, df_data, n_test_aug, mode),
            shuffle=False,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True
        )
    else:
        loader = DataLoader(
            dataset=PredictionDatasetPure(name_list, df_data, n_test_aug, mode),
            shuffle=False,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True
        )

    model.eval()
    all_outputs = []
    all_names = []
    all_truth = []
    
    all_outputs = torch.FloatTensor().cuda()
    all_truth = torch.FloatTensor().cuda()

    for names, inputs, labels in tqdm(loader, desc='Predict'):
        labels = labels.view(-1, 28).contiguous().cuda(async=True)
        all_truth = torch.cat((all_truth, labels), 0)
        with torch.no_grad():
            inputs = torch.tensor(inputs)
        outputs = model(inputs)
        all_outputs = torch.cat((all_outputs, outputs.data), 0)
        all_names.extend(names)
    datanpGT = all_truth.cpu().numpy()
    datanpPRED = all_outputs.cpu().numpy()

    return datanpPRED, all_names, datanpGT

def apply_threasholds(y_pred, threasholds):
    temp = y_pred.copy()
    for i, value in enumerate(threasholds):
        temp[:, i] = temp[:, i] > value
    return temp

# average augmented predictions
def group_aug(val_p_aug, val_names_aug, val_truth_aug):

    df_prob = pd.DataFrame(val_p_aug)
    df_prob['id'] = val_names_aug
    
    df_truth = pd.DataFrame(val_truth_aug)
    df_truth['id'] = val_names_aug

    g_prob = df_prob.groupby('id').mean()
    g_prob = g_prob.reset_index()
    g_prob = g_prob.sort_values(by='id')
    
    g_truth = df_truth.groupby('id').mean()
    g_truth = g_truth.reset_index()
    g_truth = g_truth.sort_values(by='id')

    return g_prob.drop('id', 1).values, g_truth['id'].values, g_truth.drop('id', 1).values


if __name__ == '__main__':

    with open('SETTINGS.json', 'r') as f:
        path_dict = json.load(f)

    # TTA
    valid_transform_aug = albumentations.Compose([

        albumentations.Flip(p=0.75),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=90, interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_REFLECT_101, p=1),

        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])

    csv_path = path_dict['KAGGLE_LABEL_TRAIN_DATA_CLEAN_PATH']
    df_all = pd.read_csv(csv_path)

    # load test images' name
    test_csv_path = path_dict['SUBMIT_CSV_PATH']
    df_test = pd.read_csv(test_csv_path)
    c_test = list(set(df_test['Id'].values.tolist()))

    is_aug = True
    num_aug = 5
    batch_size = 8

    # 7 trained models
    model_name_list = ['DenseNet121_change_avg_512_all_more_train_add_3_v2', 'DenseNet121_change_avg_512_all_more_train_add_3_v3', 'DenseNet121_change_avg_512_all_more_train_add_3_v5', 'DenseNet169_change_avg_512_all_more_train_add_3_v5', 'se_resnext50_32x4d_512_all_more_train_add_3_v5', 'Xception_osmr_512_all_more_train_add_3_v5', 'ibn_densenet121_osmr_512_all_more_train_add_3_v5_2']
    val_truth_oof = df_all.sort_values(by='Id').reset_index(drop = True).loc[:, 'Nucleoplasm':'Rods & rings'].values

    # predict test images on each model
    for model_name in model_name_list:
        print(model_name)
        model_snapshot_path = path_dict['MODEL_CHECKPOINT_DIR'] + model_name + '/'
        trained_model_snapshot_path = path_dict['TRAINED_MODEL_CHECKPOINT_DIR'] + model_name + '/'
        for fold in range(5):
            
            mm_name = ''
            if 'DenseNet121_change_avg' in model_name:
                mm_name = 'DenseNet121_change_avg'
            elif 'DenseNet169_change_avg' in model_name:
                mm_name = 'DenseNet169_change_avg'
            elif 'se_resnext50_32x4d' in model_name:
                mm_name = 'se_resnext50_32x4d'
            elif 'Xception_osmr' in model_name:
                mm_name = 'Xception_osmr'
            elif 'ibn_densenet121_osmr' in model_name:
                mm_name = 'ibn_densenet121_osmr'

            # K_FOLD_V2 was used to train ibn_densenet121_osmr, K_FOLD_V1 was used to train others.
            if mm_name == 'ibn_densenet121_osmr':
                kfold_path = path_dict['K_FOLD_PATH_V2']
            else:
                kfold_path = path_dict['K_FOLD_PATH_V1']
            if not os.path.exists(model_snapshot_path + 'prediction/'):
                os.makedirs(model_snapshot_path + 'prediction/')
                
            prediction_path = model_snapshot_path+'prediction/fold_{fold}'.format(fold=fold)

            # load images' name in validation dataset
            f_val = open(kfold_path + 'fold{fold}/val.txt'.format(fold=fold), 'r')
            c_val = f_val.readlines()
            f_val.close()
            c_val = [s.replace('\n', '') for s in c_val]

            # load trained model
            model = get_model(mm_name)
            model = nn.DataParallel(model).cuda()
            state = torch.load(trained_model_snapshot_path + 'model_min_loss_{fold}.pth.tar'.format(fold=fold))
            epoch = state['epoch']
            best_valid_loss = state['best_loss']
            val_f1_mean = state['val_f1_mean']
            val_threshold = state['val_threshold']
            val_f1 = state['val_f1']
            model.load_state_dict(state['state_dict'])

            print(epoch, best_valid_loss, val_f1_mean, val_threshold, val_f1)
            model.eval()
            
            # predict on validation dataset and test dataset, then save the result
            if is_aug:
                val_p_aug, val_names_aug, val_truth_aug = predict(model, c_val, df_all, batch_size, num_aug, aug=True, mode='val')
                val_predictions_aug, val_image_names_aug, val_truth_aug = group_aug(val_p_aug, val_names_aug, val_truth_aug)
                val_loss_aug = log_loss(val_truth_aug.ravel(), val_predictions_aug.ravel(), eps=1e-7)
                print('val_loss_aug = ', val_loss_aug)

                # Find threasholds aug
                max_threshold_aug, max_result_f1_aug = search_f1(val_predictions_aug, val_truth_aug)
                print('max_threshold_aug = ', max_threshold_aug, 'tuned f1_aug = ', max_result_f1_aug, np.mean(max_result_f1_aug))

                test_p_aug, test_names_aug, test_truth_aug = predict(model, c_test, df_test, batch_size, num_aug, aug=True, mode='test')
                test_predictions_aug, test_image_names_aug, test_truth_aug = group_aug(test_p_aug, test_names_aug, test_truth_aug)
                test_vote_aug = apply_threasholds(test_predictions_aug, max_threshold_aug)
                test_prob_vote_aug = np.concatenate([test_predictions_aug, test_vote_aug], 1)
                df = pd.DataFrame(test_prob_vote_aug)
                df['id'] = test_image_names_aug
                df.to_csv(prediction_path + '_test_aug_{num_aug}.csv'.format(num_aug=num_aug))
                df = pd.DataFrame(val_predictions_aug)
                df['id'] = val_image_names_aug
                df.to_csv(prediction_path + '_val_aug_{num_aug}.csv'.format(num_aug=num_aug))

            # log
            with open(model_snapshot_path + 'prediction/{model_name}.csv'.format(model_name=model_name), 'a', newline='') as f:
                writer = csv.writer(f)
                if is_aug:
                    writer.writerow([fold, num_aug, val_loss_aug, np.mean(max_result_f1_aug), max_threshold_aug, max_result_f1_aug])  

        # generate out-of-fold predictions, and save the result
        val_lists_aug = []
        test_lists_aug = []
        prediction_path = model_snapshot_path + 'prediction/'
        for fold in range(5):

            if is_aug:
                df_val_aug = pd.read_csv(prediction_path + 'fold_{fold}_val_aug_{num_aug}.csv'.format(fold=fold, num_aug=num_aug), index_col=0)
                df_test_aug = pd.read_csv(prediction_path + 'fold_{fold}_test_aug_{num_aug}.csv'.format(fold=fold, num_aug=num_aug), index_col=0)
                val_lists_aug.append(df_val_aug)
                test_lists_aug.append(df_test_aug)

        if is_aug:
            df_val_aug = pd.concat(val_lists_aug)
            df_val_aug = df_val_aug.sort_values(by='id').reset_index(drop = True)
            df_val_aug.to_csv(prediction_path + 'val_aug_{num_aug}.csv'.format(num_aug=num_aug), index=0)

            val_predictions_aug = df_val_aug.loc[:, '0':'27'].values
            val_loss_aug = log_loss(val_truth_oof.ravel(), val_predictions_aug.ravel(), eps=1e-7)
            val_threshold_aug, val_f1_aug = search_f1(val_predictions_aug, val_truth_oof)
            print(val_loss_aug, val_threshold_aug, val_f1_aug, np.mean(val_f1_aug))

        # log
        with open(model_snapshot_path + 'prediction/{model_name}.csv'.format(model_name=model_name), 'a', newline='') as f:
            writer = csv.writer(f) 
            if is_aug:
                writer.writerow(['total', val_loss_aug, val_threshold_aug, val_f1_aug, np.mean(val_f1_aug)])  

        # average test predictions
        if is_aug:
            df_test_aug = pd.concat(test_lists_aug)
            df_test_aug = df_test_aug.groupby('id').mean()
            df_test_aug.to_csv(prediction_path + 'test_aug_{num_aug}.csv'.format(num_aug=num_aug))
            test_aug_oof = df_test_aug.loc[:, '0':'27'].values

            test_vote_aug = apply_threasholds(test_aug_oof, val_threshold_aug)
            df_test_aug.loc[:, '28':'55'] = test_vote_aug
            df_test_aug.to_csv(prediction_path + 'test_aug_{num_aug}_oof.csv'.format(num_aug=num_aug))

# ensemble and submit
    MODEL_CHECKPOINT_DIR = path_dict['MODEL_CHECKPOINT_DIR']
    KAGGLE_LABEL_TRAIN_DATA_CLEAN_PATH = path_dict['KAGGLE_LABEL_TRAIN_DATA_CLEAN_PATH']
    df_all = pd.read_csv(KAGGLE_LABEL_TRAIN_DATA_CLEAN_PATH)
    val_truth_oof = df_all.sort_values(by='Id').reset_index(drop = True).loc[:, 'Nucleoplasm':'Rods & rings'].values
    models_list = [x.split('/')[-1] for x in glob(MODEL_CHECKPOINT_DIR + '*')]
    df_val_list = [pd.read_csv(MODEL_CHECKPOINT_DIR + x + '/prediction/val_aug_{num_aug}.csv'.format(num_aug=num_aug)).loc[:, '0':'27'].values for x in models_list]
    df_test_list = [pd.read_csv(MODEL_CHECKPOINT_DIR + x + '/prediction/test_aug_{num_aug}_oof.csv'.format(num_aug=num_aug)).loc[:, '0':'27'].values for x in models_list]
    val_average = sum(df_val_list)/len(df_val_list)
    max_threshold, max_result_f1 = search_f1(val_average, val_truth_oof)
    print(max_threshold, max_result_f1, np.mean(max_result_f1))
    test_aug_oof = sum(df_test_list)/len(df_test_list)

    # produce submission file 
    df_test = pd.read_csv(path_dict['SUBMIT_CSV_PATH'])
    with open(path_dict['SUBMISSION_DIR'] + 'submit_KFOLD_aug_oof_0.7_7_test.csv', 'w') as file:
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