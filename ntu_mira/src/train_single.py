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

MODEL_LABEL = 0
config.NUM_CLASS = 2

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

def get_dataflow(is_train=True):
    train_df = pd.read_csv(os.path.join('/data/kaggle/HPA', 'train.csv'))
    labels = [[int(i) for i in s.split()] for s in train_df['Target']]
    binary_label = []
    for la in labels:
        if MODEL_LABEL in la:
            binary_label.append([1])
        else:
            binary_label.append([0])

    fnames = train_df['Id'].tolist()
    fnames = [os.path.join(config.TRAIN_DATASET, f) for f in fnames]
    sprase_label = [np.eye(config.NUM_CLASS, dtype=np.float)[np.array(la)].sum(axis=0) for la in binary_label]

    if config.EXTRA:
        extra_df = pd.read_csv(os.path.join('/data/kaggle/HPA', 'HPAv18RBGY_wodpl.csv'))
        extra_labels = [[int(i) for i in s.split()] for s in extra_df['Target']]
        binary_label = []
        for la in labels:
            if MODEL_LABEL in la:
                binary_label.append([1])
            else:
                binary_label.append([0])
        extra_labels = [np.eye(config.NUM_CLASS, dtype=np.float)[np.array(la)].sum(axis=0) for la in binary_label]
        extra_fnames = extra_df['Id'].tolist()
        extra_fnames = [os.path.join(config.EXTRA_DATASET, f) for f in extra_fnames]
        fnames = fnames + extra_fnames
        sprase_label = sprase_label + extra_labels
        # extra_data = list(zip(extra_fnames, extra_labels))

    fnames = np.array(fnames)
    sprase_label = np.array(sprase_label)
    print(fnames.shape[0])
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    
    for train_index, test_index in msss.split(fnames, sprase_label):
        x_train, x_test = fnames[train_index], fnames[test_index]
        y_train, y_test = sprase_label[train_index], sprase_label[test_index]
    
    train_data = list(zip(x_train, y_train))
    val_data = list(zip(x_test, y_test))
    
    if not is_train:
        return val_data
        
    ds = DataFromList(train_data, shuffle=True)
    ds = BatchData(MapData(ds, preprocess), config.BATCH)
    ds = PrefetchDataZMQ(ds, 6)
    return ds
        
class ResnetModel(ModelDesc):
    def _get_inputs(self):
        ret = [
            InputDesc(tf.float32, (None, None, None, 4), 'image'),
            InputDesc(tf.int32, (None, config.NUM_CLASS), 'labels'),
        ]
        return ret

    def _build_graph(self, inputs):
        is_training = get_current_tower_context().is_training
        image, label = inputs
        tf.summary.image('viz', image, max_outputs=10)
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
        logits = get_logit_binary(image, num_blocks, block_func, num_class=2)
        if is_training:
            loss = softmax_cls_loss(logits, label)
            wd_cost = regularize_cost(
                '.*/W',
                l2_regularizer(1e-4), name='wd_cost')

            self.cost = tf.add_n([
                loss, wd_cost], 'total_cost')                

            add_moving_summary(self.cost, wd_cost)
        else:
            final_prob = tf.nn.softmax(logits, name="final_prob")
    
    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        print("get_nr_gpu", get_nr_gpu())
        if config.BIG:
            factor = 32 // (config.BATCH * get_nr_gpu())
            if factor != 1:
                lr = lr / float(factor)
                opt = tf.train.MomentumOptimizer(lr, 0.9)
                opt = optimizer.AccumGradOptimizer(opt, factor)
            else:
                opt = tf.train.MomentumOptimizer(lr, 0.9)
            
            #opt = tf.train.AdamOptimizer(lr)
        else:
            opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

class ResnetEvalCallback(HyperParamSetter):
    def __init__(self, param='learning_rate', mode="val_loss", is_trigger=False, patience=5, factor=0.5, base_lr=2e-3, min_lr=1e-5):
        # if a string, assumed to be a scalar graph variable
        if isinstance(param, six.string_types):
            param = GraphVarParam(param)
        assert isinstance(param, HyperParam), type(param)
        self.param = param
        self._last_value = None
        self._last_epoch_set = -1
        self.wait = 0
        self.current = 0
        self.best = -1000
        self.is_trigger = is_trigger
        self.patience = patience
        self.base_lr = base_lr
        self.factor = factor 
        self.min_lr = min_lr
        self.mode = mode # val_loss or val_f1
        
    def _setup_graph(self):
        self.param.setup_graph()
        self.pred = self.trainer.get_predictor(
            ['image'],
            get_resnet_model_output_names())
        self.valid_ds = get_dataflow(is_train=False)
        
    def _eval(self):
        from tensorpack.utils.utils import get_tqdm_kwargs

        valid_predictions = []
        valid_y = []
        th = 0.15
        with tqdm.tqdm(total=len(self.valid_ds) // config.INFERENCE_BATCH + 1, **get_tqdm_kwargs()) as pbar:
            for i in range(len(self.valid_ds) // config.INFERENCE_BATCH + 1):
                start = i * config.INFERENCE_BATCH
                end = start + config.INFERENCE_BATCH if start + config.INFERENCE_BATCH < len(self.valid_ds) else len(self.valid_ds)
                data = self.valid_ds[start:end]
                data = [preprocess(d, is_training=False) for d in data]
                x = np.array([_[0] for _ in data])
                y = np.array([_[1] for _ in data])
                final_probs = self.pred(x)
                valid_predictions.extend(np.argmax(final_probs[0], axis=-1))
                valid_y.extend(y)
                pbar.update()
        valid_predictions = np.array(valid_predictions)
        valid_y = np.array(valid_y)
        score = (valid_y == valid_predictions).mean()
        self.trainer.monitors.put_scalar("score", score)
        return score

    def _get_value_to_set(self):
        if self.current > self.best:
            self.best = self.current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.wait = 0
                current_lr = self.get_current_value()
                self.base_lr = max(current_lr * self.factor, self.min_lr)
                logger.warn("ReduceLROnPlateau reducing learning rate to {}".format(self.base_lr))
        return self.base_lr

    def _trigger_epoch(self):
        #if self.epoch_num % 10 == 0:
        self.current = self._eval()
        if self.is_trigger:
            self.trigger() # go to _get_value_to_set

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
    args = parser.parse_args()
    
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.visualize or args.evaluate or args.predict:
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
                        valid_predictions.extend(sigmoid_np(final_probs[0]))
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
            elif args.predict:
                pred = OfflinePredictor(PredictConfig(
                    model=ResnetModel(),
                    session_init=get_model_loader(args.load),
                    input_names=['image'],
                    output_names=get_resnet_model_output_names()))
                size = config.IMAGE_SIZE
                predictions = []
                test_df = pd.read_csv(os.path.join('/data/kaggle/HPA', 'sample_submission.csv'))
                fnames = test_df['Id'].tolist()
                x_test = [os.path.join(config.TEST_DATASET, f) for f in fnames]
                with tqdm.tqdm(total=(len(x_test)) // config.INFERENCE_BATCH + 1) as pbar:
                    start = 0
                    end = 0
                    for i in range(len(x_test) // config.INFERENCE_BATCH + 1):
                        start = i * config.INFERENCE_BATCH
                        end = start + config.INFERENCE_BATCH if start + config.INFERENCE_BATCH < len(x_test) else len(x_test)
                        x = x_test[start:end]
                        if (len(x) == 0):
                            break
                        if config.BIG:
                            x = np.array([open_rgby_2048(img_id) for img_id in x])    
                        else:
                            x = np.array([open_rgby(config.TEST_DATASET, img_id) for img_id in x])
                        final_probs = pred(x)
                        predictions.extend(final_probs[0])
                        pbar.update()
                    predictions = np.array(predictions)
                th = 0.15
                for th in [0.5, 0.3, 0.15]:
                    #final_pred = predictions > th
                    pp = []
                    for p in predictions:
                        if (np.all(p <= th)):
                            pp.append(p>=np.max(p))
                        else:
                            pp.append(p > th)
                    final_pred = np.array(pp)
                    final_pred = [" ".join([str(s) for s in np.arange(28)[_>0]]) for _ in final_pred]
                    df = pd.DataFrame({'Id':fnames, 'Predicted':final_pred})
                    df.to_csv("submission_{}.csv".format(th), header=True, index=False)
    else:
        logger.set_logger_dir(args.logdir)
        training_callbacks = [
            ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
            GPUUtilizationTracker(),
        ]
        if args.auto_reduce:
            stepnum = 800
            base_lr = 2e-2
            min_lr = 1e-5
            max_epoch = 400 
            TRAINING_SCHEDULE = ReduceLearningRateOnPlateau('learning_rate', 
                                        factor=0.5, patience=5, 
                                        base_lr=base_lr, min_lr=min_lr, window_size=800)
            
            training_callbacks.append(ResnetEvalCallback())
            training_callbacks.append(TRAINING_SCHEDULE)
        elif args.auto_reduce_validation:
            stepnum = 1000 
            base_lr = 3e-2
            min_lr = 1e-5
            max_epoch = 50 
            #TRAINING_SCHEDULE = ReduceLearningRateOnPlateau('learning_rate', 
            #                            factor=0.5, patience=5, 
            #                            base_lr=base_lr, min_lr=min_lr, window_size=800)
            training_callbacks.append(ResnetEvalCallback(is_trigger=True, mode="val_f1", base_lr=base_lr))
        else:
            # heuristic setting for baseline
            # 105678 train+extra
            stepnum = 20 if not config.BIG else 13210
            max_epoch = 50 

            TRAINING_SCHEDULE = ScheduledHyperParamSetter('learning_rate', [(0, 3e-2), (15, 1e-2), (30, 3e-3), (40, 1e-4)])
            
            training_callbacks.append(ResnetEvalCallback(is_trigger=False))
            training_callbacks.append(TRAINING_SCHEDULE)
                
        #==========LR Range Test===============

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