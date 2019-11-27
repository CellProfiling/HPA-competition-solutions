#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: model.py

import tensorflow as tf
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils import get_current_tower_context
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.models import ( 
    MaxPooling, BatchNorm, Conv2DTranspose, BNReLU, Conv2D, FullyConnected, GlobalAvgPooling, layer_register, Deconv2D, Dropout)
from resnet_model import (
    resnet_backbone_dropout, preresnet_group, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck, se_resnet_bottleneck,
    resnet_backbone)
import numpy as np
import config
import math
from tensorflow.python.keras.metrics import categorical_accuracy
import lovasz as L

def dice_loss(y_true, y_pred):
     epsilon_denominator = 0.001
     y_true_f = tf.reshape(y_true, [-1])
     y_pred_f = tf.reshape(y_pred, [-1])
     intersection = tf.reduce_sum(y_true_f * y_pred_f)
     score = (2. * intersection + epsilon_denominator) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + epsilon_denominator)
     return 1 - tf.reduce_mean(score)

def F1_loss(y_true, y_pred):
    epsilon_denominator = 1e-6
    beta = 1
    num_pos = tf.reduce_sum(y_pred, axis=1)
    num_true = tf.reduce_sum(y_true, axis=1)
    tp = tf.reduce_sum(y_true* y_pred, axis=1)
    precise = tp / num_pos
    recall = tp / num_true
    score = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + epsilon_denominator)
    return 1 - tf.reduce_mean(score)


def get_logit(image, num_blocks, block_func):
    with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format="NCHW"):
            return resnet_backbone_dropout(
                image, num_blocks,
                preresnet_group if config.RESNET_MODE == 'preact' else resnet_group, block_func)

def get_logit_binary(image, num_blocks, block_func, num_class=2):
    with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format="NCHW"):
            return resnet_backbone_dropout(
                image, num_blocks,
                preresnet_group if config.RESNET_MODE == 'preact' else resnet_group, block_func, num_class=num_class)

@under_name_scope()
def softmax_cls_loss(label_logits, label):
    with tf.name_scope('cls_label_metrics'):
        label_pred = tf.nn.softmax(label_logits)
        accuracy = categorical_accuracy(label, label_pred)
        add_moving_summary(accuracy)

    label_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.to_float(label), logits=label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')
    add_moving_summary(label_loss)
    return label_loss

@under_name_scope()
def cls_loss(label_logits, label):
    with tf.name_scope('cls_label_metrics'):
        label_pred = tf.nn.sigmoid(label_logits)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(label_pred), tf.to_float(label)), tf.float32), name="accuracy")
        add_moving_summary(accuracy)

    # label smoothing
    #label = tf.to_float(label)
    #label = label - (0.1 * (label - 1. / tf.cast(label.shape[-1], tf.float32)))

    if config.F1_LOSS and not config.LOVASZ:
        label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.to_float(label), logits=label_logits)
        label_loss = tf.reduce_mean(label_loss, name='label_loss')
        f1_loss = F1_loss(tf.to_float(label), tf.nn.sigmoid(label_logits))
        add_moving_summary(label_loss, f1_loss)
        return label_loss + f1_loss
    elif config.LOVASZ:
        #label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        #    labels=tf.to_float(label), logits=label_logits)
        #label_loss = tf.reduce_mean(label_loss, name='label_loss')
        lovasz = L.lovasz_softmax(tf.nn.sigmoid(label_logits), tf.to_float(tf.argmax(label, axis=1)), classes='present')
        #add_moving_summary(label_loss)
        return lovasz
    else:
        label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.to_float(label), logits=label_logits)
        label_loss = tf.reduce_mean(label_loss, name='label_loss')
        add_moving_summary(label_loss)
        return label_loss

@under_name_scope()
def weighted_loss(label_logits, label):
    with tf.name_scope('cls_label_metrics'):
        label_pred = tf.nn.sigmoid(label_logits)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(label_pred), tf.to_float(label)), tf.float32), name="accuracy")
        add_moving_summary(accuracy)

    class_weights = tf.constant([ 0.07411719,  0.761563,    0.2637393,   0.61178732,  0.51399354,  0.38002388,
                0.94742063,  0.33841247, 18.01886792, 21.22222222, 34.10714286,  0.87374199,
                1.3880814,  1.77839851,  0.89587242, 45.47619048,  1.80188679,  4.54761905,
                1.05875831,  0.64439946,  5.55232558,  0.25284617,  1.19077307,  0.32209106,
                2.96583851,  0.11606709,  2.91158537, 86.81818182])
    # turn to Batch * NUM_CLASS
    onehot_labels = tf.one_hot(label, config.NUM_CLASS) 
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)
    weights = tf.stop_gradient(weights)
    label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(label), logits=label_logits) # B*num_class

    weighted_label_loss = label_loss * weights
    weighted_label_loss = tf.reduce_mean(weighted_label_loss, name='label_loss')
    add_moving_summary(weighted_label_loss)
    return weighted_label_loss
    
            