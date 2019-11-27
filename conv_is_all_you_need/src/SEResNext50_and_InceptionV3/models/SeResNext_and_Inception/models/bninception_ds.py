# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:48:11 2018

@author: Xuan-Laptop
"""

import torch
import ResNet
import bninception
from torch import nn
from torch.nn import functional as F
#from torchvision import models
#import torchvision

class Model(nn.Module):
    def load_pretrain(self, pretrain_file="../bn_inception-52deb4733.pth"):
        self.bnInception.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))
    
    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(Model, self).__init__(**kwargs)
        
        self.bnInception = bninception.BNInception()
                
        if pretrained:
            self.load_pretrain()
        
        # change the first conv layer from 7,7,3,64 -> 7,7,4,64
        # weight of the last layer is taken as the average of the first and thrid layer
        self.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        if pretrained:
            w = self.bnInception.conv1_7x7_s2.weight
            self.conv1.weight = nn.Parameter(torch.cat((w, (w[:,:1,:,:]+w[:,2:,:,:])*0.5), dim=1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)
#        
        self.logit = nn.Sequential(
                nn.BatchNorm1d(1024),
                # nn.Dropout(0.5),
                nn.Linear(1024,28)
                )
    
    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [104, 117, 128] # for bgr
        mean = [128, 117, 104]
        std = [1, 1, 1]
        x[:, 2, :, :] = x[:, 2, :, :] - mean[2]
        x[:, 1, :, :] = x[:, 1, :, :] - mean[1]
        x[:, 0, :, :] = x[:, 0, :, :] - mean[0]
        x = x[:,[2,1,0,3],:,:]

        x = self.conv1(x)                                      #; print('x', x.size())
        x = self.bnInception.features_4(x)                     #; print('x', x.size())
        f = self.avgpool(x).view(batch_size, -1)+self.maxpool(x).view(batch_size, -1)                  #; print('f', f.size())
        logit = self.logit(f)                                  #; print('logit',logit.size())
        return logit        


if __name__ == '__main__':
    net = Model()
    net.load_pretrain()
    x = torch.rand(4, 4, 224, 224)
    net.forward(x)