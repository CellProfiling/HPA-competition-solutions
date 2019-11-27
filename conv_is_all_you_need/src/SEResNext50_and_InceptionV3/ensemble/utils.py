# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 20:46:22 2018

@author: Xuan Cao
"""
import os
import sys 
import torch
import numpy as np 
import pandas as pd
from torch import nn
import torch.nn.functional as F 
from torch.autograd import Variable

#class FocalLoss(nn.Module):
#    def __init__(self, alpha=0.25,gamma=2):
#        super(FocalLoss, self).__init__()
#        self.alpha = alpha
#        self.gamma = gamma
#
#    def forward(self, x, y):
#        '''Focal loss.
#        Args:
#          x: (tensor) sized [N,D].
#          y: (tensor) sized [N,].
#        Return:
#          (tensor) focal loss.
#        '''
#        t = Variable(y).cuda()  # [N,20]
#
#        p = x.sigmoid()
#        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
#        w = self.alpha*t + (1-self.alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
#        w = w * (1-pt).pow(self.gamma)
#        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

def encoding(x):
    # one-hot encoding
    if x == 'nan':
        y = np.zeros(28, dtype=np.float)
    else:
        labels = list(set(x.strip().split(' ')))
        labels = np.array(list(map(int, labels)))
        y = np.eye(28, dtype=np.float)[labels].sum(axis=0)
    return y

def generate_submission(labels):
    submissions = []
    for row in labels:
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    return submissions

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def np_macro_f1(y_true, y_pred, epsilon=1e-7, return_details=False):
    details = pd.DataFrame(index=list(range(28)))
    details['true_positives'] = tp = np.sum(y_true * y_pred, axis=0)
    # details['true_negatives'] = tn = np.sum((1 - y_true) * (1 - y_pred), axis=0)
    details['false_positives'] = fp = np.sum((1 - y_true) * y_pred, axis=0)
    details['false_negatives'] = fn = np.sum(y_true * (1 - y_pred), axis=0)

    details['precision'] = p = tp / (tp + fp + epsilon)
    details['recall'] = r = tp / (tp + fn + epsilon)

    out = 2 * p * r / (p + r + epsilon)
    # replace all NaN's with 0's
    details['f1_scores'] = out = np.where(np.isnan(out), np.zeros_like(out), out)
    out = np.mean(out)
    if return_details:
        return details
    else:
        return out


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)