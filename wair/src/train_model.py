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
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

#============ PyTorch imports ============#
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR

import torch.utils.data as data
from src.models.models import *
from src.dataset.dataset import *
from src.tuils.tools import *
from src.tuils.lrs_scheduler import WarmRestart, warm_restart
from src.tuils.loss_function import *
import torch.nn.functional as F

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)


# compute loss, auc, threshold, maximum f1 on validation dataset
def epochVal(model, dataLoader, optimizer, scheduler, loss):

    model.eval ()
    lossVal = 0
    lossValNorm = 0
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    
    for i, (input, target) in enumerate (dataLoader):

        target = target.view(-1, 28).contiguous().cuda(async=True)
        outGT = torch.cat((outGT, target), 0)
        with torch.no_grad():
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)    
        varOutput = model(varInput)
        losstensor = loss(varOutput, varTarget)
        outPRED = torch.cat((outPRED, varOutput.data), 0)
        lossVal += losstensor.item()
        lossValNorm += 1
    
    outLoss = lossVal / lossValNorm
    max_threshold, max_result_f1 = search_f1(outPRED, outGT)
    auc = computeAUROC(outGT, outPRED, 28)
    
    return outLoss, auc, max_threshold, max_result_f1


def train_one_model(model_name):

    snapshot_path = MODEL_CHECKPOINT_DIR + model_name
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # log
    header = ['Epoch', 'Learning rate', 'Train Loss', 'Val Loss', 'Time', 'F1 score', 'AUC'] + list(range(28)) + list(range(28)) + list(range(28))
    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # load corrected labels
    df_all = pd.read_csv(ALL_LABEL_TRAIN_DATA_CLEAN_PATH)

    for num_fold in range(5):
        print(num_fold)

        # log
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([num_fold]) 
            writer.writerow(['train_batch_size:', str(train_batch_size), 'val_batch_size:', str(val_batch_size), 'backbone', model_name, 'Image_size', Image_size])        

        # get model name
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

        # create model from model name
        model = get_model(mm_name)
        model = torch.nn.DataParallel(model).cuda()

        # get c_train and c_val used to create training and validation dataset
        f_train = open(K_FOLD_PATH_V2 + 'fold' + str(num_fold) + '/train.txt', 'r')
        f_val = open(K_FOLD_PATH_V2 + 'fold' + str(num_fold) + '/val.txt', 'r')
        c_train = f_train.readlines()
        c_val = f_val.readlines()
        f_train.close()
        f_val.close()
        c_train = [s.replace('\n', '') for s in c_train]
        c_val = [s.replace('\n', '') for s in c_val]
        
        # oversample with different strategy
        if 'v2' in model_name:
            c_train, c_val = balance_class_process_all_more_train_add_3_v2(c_train, c_val)
        if 'v3' in model_name:
            c_train, c_val = balance_class_process_all_more_train_add_3_v3(c_train, c_val)
        if 'v5' in model_name:
            c_train, c_val = balance_class_process_all_more_train_add_3_v4(c_train, c_val)

        # log
        print('train dataset:', len(c_train), '  val dataset:', len(c_val))
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['train dataset:', len(c_train), '  val dataset:', len(c_val)])  

        # create training and validation dataset
        train_loader, val_loader = generate_dataset_loader(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers)

        optimizer = torch.optim.Adamax(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = WarmRestart(optimizer, T_max=10, T_mult=1, eta_min=1e-5)

        loss = BinaryEntropyLoss_weight()

        trMaxEpoch = 24
        lossMIN = 100000
        val_f1_mean = 0
        val_auc_mean = 0

        # training 
        for epochID in range (0, trMaxEpoch):

            start_time = time.time()
            model.train()
            trainLoss = 0
            lossTrainNorm = 0

            for batchID, (input, target) in enumerate (train_loader):

                target = target.view(-1, 28).contiguous().cuda(async=True)
                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)  
                varOutput = model(varInput)
                lossvalue = loss(varOutput, varTarget)
                trainLoss = trainLoss + lossvalue.item()
                lossTrainNorm = lossTrainNorm + 1
                optimizer.zero_grad()
                lossvalue.backward()
                optimizer.step()  
            if epochID < 19:
                scheduler.step()
                scheduler = warm_restart(scheduler, T_mult=2)
            elif epochID < 21:
                optimizer = torch.optim.Adam (model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-08, amsgrad=True)    
            else:
                optimizer = torch.optim.Adam (model.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-08, amsgrad=True)

            trainLoss = trainLoss / lossTrainNorm    
            if (epochID+1)%10 == 0 or epochID > 19 or epochID == 0:
                valLoss, val_auc, val_threshold, val_f1  = epochVal(model, val_loader, optimizer, scheduler, loss)


            epoch_time = time.time() - start_time
            if valLoss < lossMIN:
                lossMIN = valLoss    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict(), 'val_threshold' : val_threshold, 'val_f1' : val_f1, 'val_f1_mean' : np.mean(val_f1), 'val_auc' : val_auc, 'val_auc_mean' : np.mean(val_auc) }, snapshot_path + '/model_min_loss_' + str(num_fold) +  '.pth.tar')

            result = [epochID, round(optimizer.state_dict()['param_groups'][0]['lr'], 5), round(trainLoss, 4), round(valLoss, 4), round(epoch_time, 0), round(np.mean(val_f1), 3), round(np.mean(val_auc), 4)]
            print(result)
            with open(snapshot_path + '/log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                
                writer.writerow(result + val_threshold + val_f1 + val_auc)  

        del model    

if __name__ == '__main__':
    with open('SETTINGS.json', 'r') as f:
        path_dict = json.load(f)

    ALL_LABEL_TRAIN_DATA_CLEAN_PATH = path_dict['ALL_LABEL_TRAIN_DATA_CLEAN_PATH']

    MODEL_CHECKPOINT_DIR = path_dict['MODEL_CHECKPOINT_DIR']
    K_FOLD_PATH_V2 = path_dict['K_FOLD_PATH_V2']

    Image_size = 512
    train_batch_size = 48
    val_batch_size = 24
    workers = 24


    backbone_lists = ['DenseNet121_change_avg_512_all_more_train_add_3_v2', 'DenseNet121_change_avg_512_all_more_train_add_3_v3', 'DenseNet121_change_avg_512_all_more_train_add_3_v5', 'DenseNet169_change_avg_512_all_more_train_add_3_v5', 'se_resnext50_32x4d_512_all_more_train_add_3_v5', 'Xception_osmr_512_all_more_train_add_3_v5', 'ibn_densenet121_osmr_512_all_more_train_add_3_v5_2']

    for backbone in backbone_lists:
        print(backbone)
        train_one_model(backbone)

