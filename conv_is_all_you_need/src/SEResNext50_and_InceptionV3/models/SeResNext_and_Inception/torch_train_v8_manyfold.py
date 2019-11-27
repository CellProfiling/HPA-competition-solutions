# add oversampling
# new image format
# inceptionresV2

import glob
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

import pandas as pd

from HPAdataset_01 import HPADataset
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils import data
from utils import FocalLoss, AverageMeter, np_macro_f1, get_learning_rate, save_checkpoint, load_checkpoint
from utils import Logger
from sklearn.metrics import f1_score
from torch.nn import functional as F

from torchvision import transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nfold',type=int,default=0,help='fold num')
parser.add_argument('--lr',type=float,default=0.001,help='initial learning rate')
parser.add_argument('--epoch',type=int, default=30, help='epoch number')
parser.add_argument('--patience', type=int, default=10, help='step scheduler patience')
parser.add_argument('--model', type=str, default='seresnext50', help='model name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--range',type=int, default=1, help='image value range')
parser.add_argument('--ds',type=int, default=1, help='whether use deep supervision')


def train(train_loader, model, criterion, optimizer, epoch, valid_loss, best_results, is_ds = 1):
    losses = AverageMeter()
    f1 = AverageMeter()
    model.train()
    for images, target in tqdm(train_loader):
        if target.shape[0]==1:
            continue
        images = images.type(torch.FloatTensor).cuda()
        target = target.type(torch.FloatTensor).cuda()
        # comput output
        if is_ds:
            output, output_64, output_128, output_256 = model(Variable(images))
            loss = ds_loss(output, output_64, output_128, output_256, target)
        else:
            output = model(Variable(images))
            loss = criterion(output, target)
        losses.update(loss.item(), images.size(0))

        f1_batch = f1_score(target.cpu().numpy(), output.sigmoid().cpu() > 0.15, average='macro')
        # f1_batch = np_macro_f1(target, output.sigmoid().cpu() > 0.15)
        f1.update(f1_batch, images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return [losses.avg, f1.avg]


def evaluate(val_loader, model, criterion, epoch, train_loss, best_results, is_ds = 1):
    losses = AverageMeter()
    f1 = AverageMeter()
    # model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images_var = images.cuda()
            target = torch.from_numpy(np.array(target)).float().cuda()
            if is_ds:
                output, _, _, _ = model(images_var)
            else:
                output = model(images_var)

            if i == 0:
                total_output = output
                total_target = target
            else:
                total_output = torch.cat([total_output, output], 0)
                total_target = torch.cat([total_target, target], 0)

        loss = criterion(total_output, total_target)
        losses.update(loss.item(), images_var.size(0))
        f1_batch = f1_score(total_target.cpu().numpy(), total_output.sigmoid().cpu().numpy() > 0.15, average='macro')
        f1.update(f1_batch, images_var.size(0))
    return [losses.avg, f1.avg]


if __name__=='__main__':

    args = parser.parse_args()

    # Some parameters
    Nfold = args.nfold
    epoch = args.epoch
    model_name = args.model
    image_range = args.range
    is_ds = args.ds
    debug = False
    #first = 50
    load = False
    learning_rate = args.lr
    if load:
        learning_rate = 0.005
    if debug:
        bat_size = 16
        img_size = 128
    else:
        bat_size = args.batch_size
        img_size = 512



    criterion = nn.BCEWithLogitsLoss().cuda()
    def ds_loss(logit, logit_64, logit_128, logit_256, label):
        loss_64 = criterion(logit_64, label)
        loss_128 = criterion(logit_128, label)
        loss_256 = criterion(logit_256, label)
        loss_final = criterion(logit, label)
        return loss_final+0.1*loss_64+0.2*loss_128+0.3*loss_256

    if model_name == 'seresnext50':
        from models.seresnext50_ds import Model
    elif model_name == 'inceptionv3':
        from models.inceptionv3_ds import Model
    elif model_name=='bninception':
        from models.bninception_ds import Model
    else:
        raise ValueError('unknown model')

    model = Model()
    if not debug:
        model = nn.DataParallel(model)  # modified by wyh
    model.cuda()

    filename = 'final_{}_ds_fold_{}'.format(model_name, str(Nfold))

    directory = '../data/'
    save_path = '../checkpoint'
    checkpoint = 'final_{}_fold_{}_lr0.01_maxEpoch{}.pth'.format(model_name, str(Nfold), str(epoch))

    log = Logger()
    log.open('../logs/log-{}'.format(filename), 'a+')
    log.write("======================\n")
    log.write("model name: %s\n"%model_name)
    log.write("learning rate:%.4f\n" % learning_rate)
    log.write("batch size: %d\n" % bat_size)
    log.write("epoch: %d\n" % epoch)
    log.write("fold No: %d\n" % Nfold)

    #set_resnet_freeze(model, True)

    start = 0
    #criterion = FocalLoss().cuda()

    best_loss = 999
    best_f1 = 0
    best_results = [np.inf, 0]
    val_metrics = [np.inf, 0]
    resume = False
    best_model = {}

    train_dfs=[]
    for i in range(5):
        if i==Nfold:
            val_images = pd.read_csv('../data/fold_%d.csv'%i)
        else:
            train_dfs.append(pd.read_csv('../data/fold_%d.csv'%i))

    train_images = pd.concat(train_dfs)
    train_images.reset_index(inplace=True,drop=True)
    train_images['ext']=train_images['group'].apply(lambda x:1 if x=='hpa_rgby' else 0)
    val_images['ext']=val_images['group'].apply(lambda x:1 if x=='hpa_rgby' else 0)
    train_path = os.path.join(directory, 'processed_train')

    # Prepare dataset
    train_df_orig=train_images.copy()
    lows = [15,15,15,15,
            8,9,10,16,
            17,20,24,26,
            27,27]
    for i in lows:
        target = str(i)
        indicies = train_df_orig.loc[train_df_orig['Target'] == target].index
        train_images = pd.concat([train_images,train_df_orig.loc[indicies]], ignore_index=True)
        indicies = train_df_orig.loc[train_df_orig['Target'].str.startswith(target+" ")].index
        train_images = pd.concat([train_images,train_df_orig.loc[indicies]], ignore_index=True)
        indicies = train_df_orig.loc[train_df_orig['Target'].str.endswith(" "+target)].index
        train_images = pd.concat([train_images,train_df_orig.loc[indicies]], ignore_index=True)
        indicies = train_df_orig.loc[train_df_orig['Target'].str.contains(" "+target+" ")].index
        train_images = pd.concat([train_images,train_df_orig.loc[indicies]], ignore_index=True)

    log.write('training set size: %d\n'%train_images.shape[0])
    log.write('validation set size: %d\n'%val_images.shape[0])

    dataset_train = HPADataset(train_path, train_images, img_size,color='rgby',
                               transforms=True, ext_path='../ext_data/moredata_compressed',range=image_range)
    dataset_val = HPADataset(train_path, val_images, img_size, mode='valid',color='rgby',
                             ext_path='../ext_data/moredata_compressed', range=image_range)

    train_loader = data.DataLoader(dataset_train, batch_size=bat_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(dataset_val, batch_size=bat_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=learning_rate)

    #scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, min_lr = 0.0001, verbose=True)
    scheduler = StepLR(optimizer, step_size=args.patience, gamma=0.1)
    train_profile = {'epoch': [], 'train_loss':[], 'val_loss':[], 'train_f1':[], 'val_f1':[]}
    for e in (range(epoch)):
        #scheduler.step(val_metrics[1])
        scheduler.step()
        # train
        lr = get_learning_rate(optimizer)
        train_metrics = train(train_loader, model, criterion, optimizer, e, val_metrics, best_results, is_ds=is_ds)
        # val
        val_metrics = evaluate(val_loader, model, criterion, e, train_metrics, best_results, is_ds=is_ds)
        #check result
        is_bset_loss = val_metrics[0] < best_results[0]
        best_results[0] = min(val_metrics[0], best_results[0])
        is_best_f1 = val_metrics[1] > best_results[1]
        best_results[1] = max(val_metrics[1], best_results[1])

        print("Epoch: %d, LR: %.5f, Train-loss: %.4f, Train-f1_macro: %.4f, Val-loss: %.4f Val-f1_macro: %.4f" % (
            e, lr, train_metrics[0], train_metrics[1], val_metrics[0], val_metrics[1]))
        log.write("Epoch: %d, LR: %.5f, Train-loss: %.4f, Train-f1_macro: %.4f, Val-loss: %.4f Val-f1_macro: %.4f\n" % (
            e, lr, train_metrics[0], train_metrics[1], val_metrics[0], val_metrics[1]))

        # Save train log
        train_profile['epoch'].append(e)
        train_profile['train_loss'].append(train_metrics[0])
        train_profile['val_loss'].append(val_metrics[0])
        train_profile['train_f1'].append(train_metrics[1])
        train_profile['val_f1'].append(val_metrics[1])

        # save the model
        if is_bset_loss:
            #best_model['model'] = model
            #best_model['opt'] = optimizer
            best_model['info'] = "Epoch: %d, Train-loss: %.4f, Train-f1_macro: %.4f, Val-loss: %.4f Val-f1_macro: %.4f" % (
            e, train_metrics[0], train_metrics[1], val_metrics[0], val_metrics[1])

            save_checkpoint(os.path.join(save_path, '{}_lr{}_maxEpoch{}.pth'.format(filename, learning_rate, epoch)), model,
                            optimizer)