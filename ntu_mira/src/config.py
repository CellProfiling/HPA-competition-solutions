#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py

import numpy as np

# for new data
NORM = 'BN' 
FOLD = 4
RGB = False
PSEUDO = True
WEIGHTED_SAMPLE = False
OVERSAMPLE = True
LOVASZ = False
F1_LOSS = False
ACC = False
BIG = True
EXTRA = True
DROPOUT = True
RESNET_MODE = "resnet" #preact, se, resnet
FREEZE = False
FREEZE_BN = False
BATCH = 16 if not BIG else 8
INFERENCE_BATCH = 1 if not BIG else 1
IMAGE_SIZE = 1024
NUM_CLASS = 28
RESNET_DEPTH = 34
if BIG and RESNET_DEPTH == 34:
    BATCH = 16
    INFERENCE_BATCH = 1
if BIG and RESNET_DEPTH == 18:
    BATCH = 16
    INFERENCE_BATCH = 1
if BIG and RESNET_DEPTH == 34 and ACC:
    BATCH = 4
    INFERENCE_BATCH = 1
RESNET = "Resnet" #ResXt
# dataset -----------------------
BASEDIR = '/data/kaggle/HPA'

TRAIN_2048 = "/data/kaggle/HPA/train_2048_pkl"
EXTRA_2048 = "/data/kaggle/HPA/HPAv18/2048_pkl"
TEST_2048 = "/data/kaggle/HPA/test_2048_pkl"

EXTRA_DATASET = "/data/kaggle/HPA/HPAv18/512" if not BIG else EXTRA_2048
TRAIN_DATASET = "/data/kaggle/HPA/train_512" if not BIG else TRAIN_2048
TEST_DATASET = "/data/kaggle/HPA/test_512" if not BIG else TEST_2048
