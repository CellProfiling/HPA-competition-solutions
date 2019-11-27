#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
import matplotlib
matplotlib.use('Agg')
import os
import argparse
import cv2
import shutil
import itertools
import tqdm
import math
import numpy as np
import json
import tensorflow as tf
import zipfile
import pickle
import six
from glob import glob

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
from tensorpack.utils import logger
import tensorpack.utils.viz as tpviz
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow import (
    DataFlow, RNGDataFlow, DataFromGenerator, MapData, imgaug, AugmentImageComponent, TestDataSpeed, MultiProcessMapData,
    MapDataComponent, DataFromList, PrefetchDataZMQ, BatchData)

from resnet_model import (
    preresnet_group, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck, se_resnet_bottleneck,
    resnet_backbone)
from basemodel import (
    image_preprocess, pretrained_resnet_conv4, resnet_conv5)
from model import *
import config
import collections
import ast
import pandas as pd
from utils import *
from tensorflow.python.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow import keras
from custom_utils import ReduceLearningRateOnPlateau
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
import scipy.optimize as opt

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

def tf_ce(x, z):
    zeros = np.zeros_like(x)
    cond = (x >= zeros)
    relu_logits = np.where(cond, x, zeros)
    neg_abs_logits = np.where(cond, -x, x)
    return np.mean(relu_logits - x * z + np.log1p(np.exp(neg_abs_logits)))

def get_batch_factor():
    nr_gpu = get_nr_gpu()
    assert nr_gpu in [1, 2, 4, 8], nr_gpu
    return 8 // nr_gpu

def get_resnet_model_output_names():
    return ['final_probs']

def oversample_2(data):
    low = np.array([8,9,10,15,17,20,26,24,27])
    data_low = []
    for d in data:
        true_label = np.arange(config.NUM_CLASS)[d[1]>0]
        if np.any([a in low for a in true_label]):
            data_low.append(d)
            data_low.append(d)
            data_low.append(d)
    return data + data_low

def oversample(df):
    df_orig = df.copy()    
    lows = [15,15,15,8,9,10,8,9,10,8,9,10,17,20,24,26,15,27,15,20,24,17,8,15,27,27,27]
    for i in lows:
        target = str(i)
        indicies = df_orig.loc[df_orig['Target'] == target].index
        df = pd.concat([df,df_orig.loc[indicies]], ignore_index=True)
        indicies = df_orig.loc[df_orig['Target'].str.startswith(target+" ")].index
        df = pd.concat([df,df_orig.loc[indicies]], ignore_index=True)
        indicies = df_orig.loc[df_orig['Target'].str.endswith(" "+target)].index
        df = pd.concat([df,df_orig.loc[indicies]], ignore_index=True)
        indicies = df_orig.loc[df_orig['Target'].str.contains(" "+target+" ")].index
        df = pd.concat([df,df_orig.loc[indicies]], ignore_index=True)
    return df
    
def get_dataflow(is_train=True):
    train_df = pd.read_csv(os.path.join('/data/kaggle/HPA', 'train.csv'))
    #train_df = oversample(train_df)
    labels = [[int(i) for i in s.split()] for s in train_df['Target']]
    fnames = train_df['Id'].tolist()
    fnames = [os.path.join(config.TRAIN_DATASET, f) for f in fnames]
    sprase_label = [np.eye(config.NUM_CLASS, dtype=np.float)[np.array(la)].sum(axis=0) for la in labels]

    extra_df = pd.read_csv(os.path.join('/data/kaggle/HPA', 'HPAv18RGBY_WithoutUncertain_wodpl.csv'))
    #extra_df = oversample(extra_df)
    extra_labels = [[int(i) for i in s.split()] for s in extra_df['Target']]
    extra_labels = [np.eye(config.NUM_CLASS, dtype=np.float)[np.array(la)].sum(axis=0) for la in extra_labels]
    extra_fnames = extra_df['Id'].tolist()
    extra_fnames = [os.path.join(config.EXTRA_DATASET, f) for f in extra_fnames]
    fnames = fnames + extra_fnames
    sprase_label = sprase_label + extra_labels
   
    fnames = np.array(fnames)
    sprase_label = np.array(sprase_label)
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    
    for train_index, test_index in msss.split(fnames, sprase_label):
        x_train, x_test = fnames[train_index], fnames[test_index]
        y_train, y_test = sprase_label[train_index], sprase_label[test_index]
    
    holdout_data = list(zip(x_test, y_test))
    # 5 fold the rest
    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=1)
    for fold_num, (train_index, test_index) in enumerate(mskf.split(x_train, y_train)):
        if fold_num == config.FOLD:
            foldx_train, foldx_test = x_train[train_index], x_train[test_index]
            foldy_train, foldy_test = y_train[train_index], y_train[test_index]
            break

    train_data = list(zip(foldx_train, foldy_train))
    val_data = list(zip(foldx_test, foldy_test))

    train_data = oversample_2(train_data)
    
    pseudo_df = pd.read_csv(os.path.join('/data/kaggle/HPA', 'LB623.csv'))
    pseudo_fnames = pseudo_df['Id'].tolist()
    pseudo_fnames = [os.path.join(config.TEST_DATASET, f) for f in pseudo_fnames]
    #pseudo_labels = np.load("./SOTA.npy")
    #pseudo_labels = [np.array(_) for _ in pseudo_labels]
    pseudo_labels = [[int(i) for i in s.split()] for s in pseudo_df['Predicted']]
    pseudo_labels = [np.eye(config.NUM_CLASS, dtype=np.float)[np.array(la)].sum(axis=0) for la in pseudo_labels]
    pseudo_data = list(zip(pseudo_fnames, pseudo_labels))
    train_data = train_data + pseudo_data
    
    print("train: ", len(train_data), len(val_data))

    if not is_train:
        return val_data
    
    ds = DataFromList(train_data, shuffle=True)
    ds = BatchData(MapData(ds, preprocess), config.BATCH)
    ds = PrefetchDataZMQ(ds, 6)
    return ds
        
class ResnetModel(ModelDesc):
    def _get_inputs(self):
        if config.RGB:
            ret = [
                InputDesc(tf.float32, (None, None, None, 3), 'image'),
                InputDesc(tf.float32, (None, config.NUM_CLASS), 'labels'),
            ]
        else:
            ret = [
                InputDesc(tf.float32, (None, None, None, 4), 'image'),
                InputDesc(tf.float32, (None, config.NUM_CLASS), 'labels'),
            ]
        return ret

    def _build_graph(self, inputs):
        is_training = get_current_tower_context().is_training
        image, label = inputs
        #tf.summary.image('viz', image, max_outputs=10)
        image = image_preprocess(image, bgr=False)
        #image = image * (1.0 / 255)
        image = tf.transpose(image, [0, 3, 1, 2])
        
        depth = config.RESNET_DEPTH
        basicblock = preresnet_basicblock if config.RESNET_MODE == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'se': se_resnet_bottleneck}[config.RESNET_MODE]
        num_blocks, block_func = {
            18: ([2, 2, 2, 2], basicblock),
            26: ([2, 2, 2, 2], bottleneck),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]
        logits = get_logit(image, num_blocks, block_func)
        if is_training:
            loss = cls_loss(logits, label)
            #wd_cost = regularize_cost(
            #    '.*/W',
            #    l2_regularizer(1e-4), name='wd_cost')

            self.cost = tf.add_n([
                loss], 'total_cost')                

            #add_moving_summary(self.cost)
        else:
            final_probs = tf.nn.sigmoid(logits, name="final_probs")
    
    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        print("get_nr_gpu", get_nr_gpu())
        if config.BIG:
            if config.ACC:
                factor = 4
                lr = lr / float(factor)
                opt = tf.train.AdamOptimizer(lr)
                opt = optimizer.AccumGradOptimizer(opt, factor)
            else:
                opt = tf.train.AdamOptimizer(lr, 0.9)
            
        else:
            #opt = tf.train.MomentumOptimizer(lr, 0.9)
            opt = tf.train.AdamOptimizer(lr)
        return opt

class ResnetEvalCallbackSimple(Callback):
        
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'],
            get_resnet_model_output_names())
        self.valid_ds = get_dataflow(is_train=False)
        
    def _eval(self):
        from tensorpack.utils.utils import get_tqdm_kwargs
        valid_predictions = []
        valid_y = []
        valid_logits = []
        th = 0.15
        total_run = len(self.valid_ds) // config.INFERENCE_BATCH
        total_run = total_run + 1 if len(self.valid_ds) % config.INFERENCE_BATCH !=0 else total_run
        with tqdm.tqdm(total=total_run, **get_tqdm_kwargs()) as pbar:
            for i in range(total_run):
                start = i * config.INFERENCE_BATCH
                end = start + config.INFERENCE_BATCH if start + config.INFERENCE_BATCH < len(self.valid_ds) else len(self.valid_ds)
                data = self.valid_ds[start:end]
                data = [preprocess(d, is_training=False) for d in data]
                x = np.array([_[0] for _ in data])
                y = np.array([_[1] for _ in data])
                if len(x) == 0:
                    break
                final_probs = self.pred(x)
                valid_predictions.extend(final_probs[0])
                valid_logits.extend(final_probs[0])
                valid_y.extend(y)
                #score += mapk(la, final_labels)
                pbar.update()
        valid_predictions = np.array(valid_predictions)
        valid_y = np.array(valid_y)
        valid_logits = np.array(valid_logits)
        val_loss = tf_ce(valid_logits, valid_y)
        F1_score_05 = calc_macro_f1(valid_predictions, valid_y, 0.5)
        F1_score_015 = calc_macro_f1(valid_predictions, valid_y, 0.15)
        F1_score_02 = calc_macro_f1(valid_predictions, valid_y, 0.2)
        print('F1_score: {:.5f} {:.5f} {:.5f}'.format(F1_score_05, F1_score_015, F1_score_02))
        self.trainer.monitors.put_scalar("F1_score", F1_score_015)
        return F1_score_015

    def _trigger_epoch(self):
        interval = 10 if config.BIG else 5
        if self.epoch_num % interval == 0:
            self._eval() # go to _get_value_to_s

def flip_trans(im):
    im = np.fliplr(im)
    im = np.transpose(im, [1,0,2])
    im = np.fliplr(im)
    return im

def inference(pred, x_test, tta=['fliplr', 'rot90'], mode='test'):
     with tqdm.tqdm(total=(len(x_test)) // config.INFERENCE_BATCH + 1) as pbar:
        start = 0
        end = 0
        predictions = []
        final_probs_tta = {}
        for i in range(len(x_test) // config.INFERENCE_BATCH + 1):
            start = i * config.INFERENCE_BATCH
            end = start + config.INFERENCE_BATCH if start + config.INFERENCE_BATCH < len(x_test) else len(x_test)
            x = x_test[start:end]
            if (len(x) == 0):
                break
            if mode == 'test':
                if config.BIG:
                    x = np.array([open_rgby_2048(img_id) for img_id in x])    
                else:
                    x = np.array([open_rgby(config.TEST_DATASET, img_id) for img_id in x])
            else:
                x = [preprocess(d, is_training=False) for d in x]
                x = np.array([_[0] for _ in x])
            final_probs = pred(x)
            predictions.extend(final_probs[0])
            if not tta:
                pbar.update()
                continue

            for k in tta:
                if k not in final_probs_tta:
                    final_probs_tta[k] = []
                if k == 'fliplr':
                    x_prime = np.array([np.fliplr(_x) for _x in x])
                    final_probs = pred(x_prime)
                elif k == 'flipud':
                    x_prime = np.array([np.flipud(_x) for _x in x])
                    final_probs = pred(x_prime)
                elif k == 'rot90':
                    x_prime = np.array([np.rot90(_x) for _x in x])
                    final_probs = pred(x_prime)
                elif k == 'rot180':
                    x_prime = np.array([np.rot90(_x, 2) for _x in x])
                    final_probs = pred(x_prime)
                elif k == 'rot270':
                    x_prime = np.array([np.rot90(_x, 3) for _x in x])
                    final_probs = pred(x_prime)
                elif k == 'transpose':
                    x_prime = np.array([np.transpose(_x, [1, 0, 2]) for _x in x])
                    final_probs = pred(x_prime)
                elif k == 'transpose_lr':
                    x_prime = np.array([flip_trans(_x) for _x in x])
                    final_probs = pred(x_prime)
                final_probs_tta[k].extend(final_probs[0])
            pbar.update()
        predictions = np.array(predictions)
        if not tta:
            return predictions
        for k in tta:
            final_probs_tta[k] = np.array(final_probs_tta[k])
        return predictions, final_probs_tta

def TTA(pred, x_test, mode='test', name=''):
    #tta = ['fliplr', 'flipud', 'rot90']
    tta = ['fliplr', 'flipud', 'rot90', 'rot180', 'rot270', 'transpose', 'transpose_lr']
    prob, probtta = inference(pred, x_test, tta=tta, mode=mode)
    
    # average ensemble
    tta_prob = [probtta[k] for k in tta] + [prob]
    model = 'resnet{}'.format(config.RESNET_DEPTH)
    size = '1024' if config.BIG else '512'
    model_name = "{}_{}_fold{}_{}".format(model, size, config.FOLD, name)
    np.save(model_name, tta_prob)
    averaged_prob = np.mean(tta_prob, axis=0)
    return averaged_prob

def Ensemble_2(fnames, mode='test'):
    ensembled = []
    config.RESNET_DEPTH = 34
    config.BIG = True
    config.RGB = True
    config.BATCH = 16
    config.INFERENCE_BATCH = 1
    config.FOLD = 0
    config.TEST_DATASET = "/data/kaggle/HPA/test_2048_pkl"
    x_test = [os.path.join(config.TEST_DATASET, f) for f in fnames]
    model_path=["./train_log/5fold/resnet34/fold0_1024/model-58833"]
    resnet34_1024_pred_func = lambda x: OfflinePredictor(PredictConfig(
                    model=ResnetModel(),
                    session_init=get_model_loader(x),
                    input_names=['image'],
                    output_names=get_resnet_model_output_names()))
    preds = [resnet34_1024_pred_func(p) for p in model_path]
    for idx, p in enumerate(preds):
        probs = TTA(p, x_test, mode=mode, name=model_path[idx].split("-")[-1])
        #probs = inference(p, x_test, tta=False, mode='test')
        #np.save('resnet34_fold0_{}'.format(model_path[idx].split("-")[-1]), probs)
        ensembled.append(probs)

    config.RESNET_DEPTH = 34
    config.BIG = True
    config.RGB = False
    config.BATCH = 16
    config.FOLD = 1
    config.INFERENCE_BATCH = 1
    config.TEST_DATASET = "/data/kaggle/HPA/test_2048_pkl"
    x_test = [os.path.join(config.TEST_DATASET, f) for f in fnames]
    model_path=["./train_log/5fold/resnet34/fold1_1024_norgb_sm/model-63191"]
    resnet34_1024_pred_func = lambda x: OfflinePredictor(PredictConfig(
                    model=ResnetModel(),
                    session_init=get_model_loader(x),
                    input_names=['image'],
                    output_names=get_resnet_model_output_names()))
    preds = [resnet34_1024_pred_func(p) for p in model_path]
    for idx, p in enumerate(preds):
        probs = TTA(p, x_test, mode=mode)
        #probs = inference(p, x_test, tta=False, mode='test')
        #np.save('resnet34_fold1_{}'.format(model_path[idx].split("-")[-1]), probs)
        ensembled.append(probs)

    config.RESNET_DEPTH = 34
    config.BIG = True
    config.RGB = False
    config.BATCH = 16
    config.INFERENCE_BATCH = 1
    config.FOLD = 2
    config.TEST_DATASET = "/data/kaggle/HPA/test_2048_pkl"
    x_test = [os.path.join(config.TEST_DATASET, f) for f in fnames]
    model_path=["./train_log/5fold/resnet34/fold2_1024/model-63191"]
    resnet34_1024_pred_func = lambda x: OfflinePredictor(PredictConfig(
                    model=ResnetModel(),
                    session_init=get_model_loader(x),
                    input_names=['image'],
                    output_names=get_resnet_model_output_names()))
    preds = [resnet34_1024_pred_func(p) for p in model_path]
    for idx, p in enumerate(preds):
        probs = TTA(p, x_test, mode=mode, name=model_path[idx].split("-")[-1])
        #probs = inference(p, x_test, tta=False, mode='test')
        #np.save('resnet34_fold1_{}'.format(model_path[idx].split("-")[-1]), probs)
        ensembled.append(probs)

    config.RESNET_DEPTH = 34
    config.BIG = True
    config.RGB = False
    config.BATCH = 16
    config.INFERENCE_BATCH = 1
    config.FOLD = 3
    config.TEST_DATASET = "/data/kaggle/HPA/test_2048_pkl"
    x_test = [os.path.join(config.TEST_DATASET, f) for f in fnames]
    model_path=["./train_log/5fold/resnet34/fold3_1024/model-63191"]
    resnet34_1024_pred_func = lambda x: OfflinePredictor(PredictConfig(
                    model=ResnetModel(),
                    session_init=get_model_loader(x),
                    input_names=['image'],
                    output_names=get_resnet_model_output_names()))
    preds = [resnet34_1024_pred_func(p) for p in model_path]
    for idx, p in enumerate(preds):
        probs = TTA(p, x_test, mode=mode, name=model_path[idx].split("-")[-1])
        #probs = inference(p, x_test, tta=False, mode='test')
        #np.save('resnet34_fold1_{}'.format(model_path[idx].split("-")[-1]), probs)
        ensembled.append(probs)

    config.RESNET_DEPTH = 34
    config.BIG = True
    config.RGB = False
    config.BATCH = 16
    config.INFERENCE_BATCH = 1
    config.FOLD = 4
    config.TEST_DATASET = "/data/kaggle/HPA/test_2048_pkl"
    x_test = [os.path.join(config.TEST_DATASET, f) for f in fnames]
    model_path=["./train_log/5fold/resnet34/fold4_1024/model-47938"]
    resnet34_1024_pred_func = lambda x: OfflinePredictor(PredictConfig(
                    model=ResnetModel(),
                    session_init=get_model_loader(x),
                    input_names=['image'],
                    output_names=get_resnet_model_output_names()))
    preds = [resnet34_1024_pred_func(p) for p in model_path]
    for idx, p in enumerate(preds):
        probs = TTA(p, x_test, mode=mode, name=model_path[idx].split("-")[-1])
        #probs = inference(p, x_test, tta=False, mode='test')
        #np.save('resnet34_fold1_{}'.format(model_path[idx].split("-")[-1]), probs)
        ensembled.append(probs)
    
    return np.mean(ensembled, axis=0)

def Ensemble_Model(fnames, mode='test'):
    ensembled = []
    config.RESNET_DEPTH = 50
    config.BIG = True
    config.BATCH = 8
    config.INFERENCE_BATCH = 1
    config.TEST_DATASET = "/data/kaggle/HPA/test_2048_pkl"
    x_test = [os.path.join(config.TEST_DATASET, f) for f in fnames]
    model_path=["./train_log/resnet50_1024/model-191545",
                "./train_log/resnet50_1024/model-198150"]
    resnet50_pred_func = lambda x: OfflinePredictor(PredictConfig(
                    model=ResnetModel(),
                    session_init=get_model_loader(x),
                    input_names=['image'],
                    output_names=get_resnet_model_output_names()))
    preds = [resnet50_pred_func(p) for p in model_path]
    for idx, p in enumerate(preds):
        probs = TTA(p, x_test, mode=mode)
        np.save('resnet50_1024_{}'.format(model_path[idx].split("-")[-1]), probs)
        ensembled.append(probs)

    config.RESNET_DEPTH = 50 # LB 0.595
    config.BIG = False
    config.BATCH = 16
    config.INFERENCE_BATCH = 16
    config.TEST_DATASET = "/data/kaggle/HPA/test_512"
    x_test = [os.path.join(config.TEST_DATASET, f) for f in fnames]
    model_path=["./train_log/resnet50_extra/model-85878",
                "./train_log/resnet50_extra/model-79272"]
    resnet50_pred_func = lambda x: OfflinePredictor(PredictConfig(
                    model=ResnetModel(),
                    session_init=get_model_loader(x),
                    input_names=['image'],
                    output_names=get_resnet_model_output_names()))
    preds = [resnet50_pred_func(p) for p in model_path]
    for idx, p in enumerate(preds):
        probs = TTA(p, x_test, mode=mode)
        np.save('resnet50_{}'.format(model_path[idx].split("-")[-1]), probs)
        ensembled.append(probs)
    config.RESNET_DEPTH = 34
    config.BIG = False
    config.BATCH = 16
    config.INFERENCE_BATCH = 16
    config.TEST_DATASET = "/data/kaggle/HPA/test_512"
    x_test = [os.path.join(config.TEST_DATASET, f) for f in fnames]
    model_path=["./train_log/resnet34_extra/model-95787",
                "./train_log/resnet34_extra/model-62757"]
    resnet34_pred_func = lambda x: OfflinePredictor(PredictConfig(
                    model=ResnetModel(),
                    session_init=get_model_loader(x),
                    input_names=['image'],
                    output_names=get_resnet_model_output_names()))
    preds = [resnet34_pred_func(p) for p in model_path]
    for idx, p in enumerate(preds):
        probs = TTA(p, x_test, mode=mode)
        np.save('resnet34_{}'.format(model_path[idx].split("-")[-1]), probs)
        ensembled.append(probs)
    return np.mean(ensembled, axis=0)

def submit(predictions, fnames):
    for th in [0.2, 0.3, 0.15, 0.12]:
        pp = []
        for p in predictions:
            if (np.all(p <= th)):
                pp.append(p>=np.max(p))
            else:
                pp.append(p > th)
        final_pred = np.array(pp)
        final_pred = [" ".join([str(s) for s in np.arange(28)[_>0]]) for _ in final_pred]
        df = pd.DataFrame({'Id':fnames, 'Predicted':final_pred})
        if config.BIG:
            df.to_csv("submission_1024_{}.csv".format(th), header=True, index=False)
        else:
            df.to_csv("submission_{}.csv".format(th), header=True, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--logdir', help='logdir', default='train_log/fastrcnn')
    parser.add_argument('--datadir', help='override config.BASEDIR')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--evaluate', help='path to the output json eval file')
    parser.add_argument('--predict', help='path to the input image file')
    parser.add_argument('--lr_find', action='store_true')
    parser.add_argument('--cyclic', action='store_true')
    parser.add_argument('--auto_reduce', action='store_true')
    parser.add_argument('--auto_reduce_validation', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    args = parser.parse_args()
    
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.visualize or args.evaluate or args.predict or args.ensemble:
        # autotune is too slow for inference
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

        assert args.load

        if args.visualize:
            visualize(args.load)
#            imgs = [img['file_name'] for img in imgs]
#            predict_many(pred, imgs)
        else:
            if args.evaluate:
                pred = OfflinePredictor(PredictConfig(
                    model=ResnetModel(),
                    session_init=get_model_loader(args.load),
                    input_names=['image'],
                    output_names=get_resnet_model_output_names()))
        
                from tensorpack.utils.utils import get_tqdm_kwargs
                valid_ds = get_dataflow(is_train=False)
                valid_predictions = []
                valid_y = []
                valid_logits = []
                th = 0.5
                total_run = len(valid_ds) // config.INFERENCE_BATCH
                total_run = total_run + 1 if len(valid_ds) % config.INFERENCE_BATCH !=0 else total_run
                with tqdm.tqdm(total=total_run, **get_tqdm_kwargs()) as pbar:
                    for i in range(total_run):
                        start = i * config.INFERENCE_BATCH
                        end = start + config.INFERENCE_BATCH if start + config.INFERENCE_BATCH < len(valid_ds) else len(valid_ds)
                        data = valid_ds[start:end]
                        data = [preprocess(d, is_training=False) for d in data]
                        x = np.array([_[0] for _ in data])
                        y = np.array([_[1] for _ in data])
                        
                        final_probs = pred(x)
                        valid_predictions.extend(final_probs[0])
                        valid_logits.extend(final_probs[0])
                        valid_y.extend(y)
                        pbar.update()
                valid_predictions = np.array(valid_predictions)
                valid_y = np.array(valid_y)
                valid_logits = np.array(valid_logits)
                np.save("pred.npy", valid_predictions)
                np.save("gt.npy", valid_y)
                val_loss = tf_ce(valid_logits, valid_y)
                print("Val Loss: ", val_loss)
                f1_score1 = calc_macro_f1(valid_predictions, valid_y, threshold=0.5)
                f1_score2 = calc_macro_f1(valid_predictions, valid_y, threshold=0.15)
                f1_score3 = calc_macro_f1(valid_predictions, valid_y, threshold=0.3)
                f1_score4 = calc_macro_f1(valid_predictions, valid_y, threshold=0.1)
                #f1_score = f1_score(valid_y.reshape(-1), (valid_predictions>th).astype(np.uint8).reshape(-1), average="macro")
                print('F1_score 0.5: {:.5f}'.format(f1_score1))
                print('F1_score 0.15: {:.5f}'.format(f1_score2))
                print('F1_score 0.3: {:.5f}'.format(f1_score3))
                print('F1_score 0.1: {:.5f}'.format(f1_score4))
                
            elif args.ensemble:
                pred = OfflinePredictor(PredictConfig(
                    model=ResnetModel(),
                    session_init=get_model_loader(args.load),
                    input_names=['image'],
                    output_names=get_resnet_model_output_names()))
                test_df = pd.read_csv(os.path.join('/data/kaggle/HPA', 'sample_submission.csv'))
                fnames = test_df['Id'].tolist()
                x_test = [os.path.join(config.TEST_DATASET, f) for f in fnames]
                #predictions = TTA(pred, x_test, mode='test')
                #predictions = Ensemble(x_test, mode='test')
                #predictions = Ensemble_Model(fnames, mode='test')
                predictions = Ensemble_2(fnames, mode='test')
                submit(predictions, fnames)

            elif args.predict:
                pred = OfflinePredictor(PredictConfig(
                    model=ResnetModel(),
                    session_init=get_model_loader(args.load),
                    input_names=['image'],
                    output_names=get_resnet_model_output_names()))
                predictions = []
                test_df = pd.read_csv(os.path.join('/data/kaggle/HPA', 'sample_submission.csv'))
                fnames = test_df['Id'].tolist()
                x_test = [os.path.join(config.TEST_DATASET, f) for f in fnames]
                #predictions = TTA(pred, x_test, mode='test', name="58833")
                #predictions = Ensemble(x_test, mode='test')
                #predictions = Ensemble_Model(x_test, mode='test')
                predictions = inference(pred, x_test, tta=False, mode='test')
                submit(predictions, fnames)
                
    else:
        train_df = pd.read_csv(os.path.join('/data/kaggle/HPA', 'train.csv'))
        num_training = len(train_df)
        if config.EXTRA:
            extra_df = pd.read_csv(os.path.join('/data/kaggle/HPA', 'HPAv18RGBY_WithoutUncertain_wodpl.csv'))
            num_training += len(extra_df)

        num_training = int(num_training * 0.85 * 0.8)
        print("num_training", num_training)

        logger.set_logger_dir(args.logdir)
        training_callbacks = [
            ModelSaver(max_to_keep=100, keep_checkpoint_every_n_hours=1),
            GPUUtilizationTracker(),
        ]
        # heuristic setting for baseline
        # 105678 train+extra
            
        stepnum = num_training // (config.BATCH * get_nr_gpu())  + 1
        max_epoch = 50
        
        if config.FREEZE:
            max_epoch = 4
            TRAINING_SCHEDULE = ScheduledHyperParamSetter('learning_rate', [(0, 1e-3)])                
        else:
            max_epoch = 25 #35
            TRAINING_SCHEDULE = ScheduledHyperParamSetter('learning_rate', [(0, 1e-5), (15, 1e-5), (25, 1e-6)])
        training_callbacks.append(ResnetEvalCallbackSimple())
        training_callbacks.append(TRAINING_SCHEDULE)
                

        cfg = TrainConfig(
            model=ResnetModel(),
            data=QueueInput(get_dataflow()),
            callbacks=training_callbacks,
            steps_per_epoch=stepnum,
            max_epoch=max_epoch,
            session_init=get_model_loader(args.load) if args.load else None,
        )
        trainer = SyncMultiGPUTrainerReplicated(get_nr_gpu(), mode='nccl')
        launch_train_with_config(cfg, trainer)