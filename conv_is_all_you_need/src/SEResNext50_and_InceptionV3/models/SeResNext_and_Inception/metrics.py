# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 13:44:53 2018

@author: Xuan-Laptop
"""

import numpy as np

def iou_kaggle(img_pred, img_truth, threshold=0.5):
    
    N = len(img_pred)
    predict = img_pred.reshape(N, -1)
    truth   = img_truth.reshape(N, -1)
    
    predict = predict > threshold
    truth = truth > threshold
    
    intersection = truth & predict
    union        = truth | predict
    
    intersection = np.sum(intersection)
    union = np.sum(union)
    if union == 0:
        union = 1e-09
    iou = intersection/union
    
    #---------------------------------------
    result = []
    precision = []
    is_empty_truth = (truth.sum(1) == 0)
    is_empty_pred  = (predict.sum(1) == 0)
    
    thresholds = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    for t in thresholds:
        p = iou >= t
        
        tp = (~is_empty_truth)  &  (~is_empty_pred)  &  (iou > t)
        fp = (~is_empty_truth)  &  (~is_empty_pred)  &  (iou <= t)
        fn = (~is_empty_truth)  &  ( is_empty_pred)
        fp_empty = ( is_empty_truth)  &  (~is_empty_pred)
        tn_empty = ( is_empty_truth)  &  ( is_empty_pred)
        
        p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)
        
        result.append( np.column_stack((tp, fp, fn, tn_empty, fp_empty)))
        precision.append(p)
        
    result = np.array(result).transpose(1,2,0)
    precision = np.column_stack(precision)
    precision = precision.mean(1)
    
    return precision, result, threshold
        
    
    