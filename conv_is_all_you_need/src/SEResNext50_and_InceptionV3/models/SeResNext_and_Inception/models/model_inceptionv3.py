# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:48:11 2018

@author: Xuan-Laptop
"""

import torch
import ResNet
import types
# from .torchvision_models import *
from torch import nn
from torch.nn import functional as F
from torchvision import models
#import torchvision

# def inceptionv3(num_classes=1000, pretrained='imagenet'):
#     r"""Inception v3 model architecture from
#     `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
#     """
#     model = models.inception_v3(pretrained=False)
#     if pretrained is not None:
#         settings = pretrained_settings['inceptionv3'][pretrained]
#         model = load_pretrained(model, num_classes, settings)
#
#     # Modify attributs
#     model.last_linear = model.fc
#     del model.fc
#
#     def features(self, input):
#         # 299 x 299 x 3
#         x = self.Conv2d_1a_3x3(input) # 149 x 149 x 32
#         x = self.Conv2d_2a_3x3(x) # 147 x 147 x 32
#         x = self.Conv2d_2b_3x3(x) # 147 x 147 x 64
#         x = F.max_pool2d(x, kernel_size=3, stride=2) # 73 x 73 x 64
#         x = self.Conv2d_3b_1x1(x) # 73 x 73 x 80
#         x = self.Conv2d_4a_3x3(x) # 71 x 71 x 192
#         x = F.max_pool2d(x, kernel_size=3, stride=2) # 35 x 35 x 192
#         x = self.Mixed_5b(x) # 35 x 35 x 256
#         x = self.Mixed_5c(x) # 35 x 35 x 288
#         x = self.Mixed_5d(x) # 35 x 35 x 288
#         x = self.Mixed_6a(x) # 17 x 17 x 768
#         x = self.Mixed_6b(x) # 17 x 17 x 768
#         x = self.Mixed_6c(x) # 17 x 17 x 768
#         x = self.Mixed_6d(x) # 17 x 17 x 768
#         x = self.Mixed_6e(x) # 17 x 17 x 768
#         if self.training and self.aux_logits:
#             self._out_aux = self.AuxLogits(x) # 17 x 17 x 768
#         x = self.Mixed_7a(x) # 8 x 8 x 1280
#         x = self.Mixed_7b(x) # 8 x 8 x 2048
#         x = self.Mixed_7c(x) # 8 x 8 x 2048
#         return x
#


class HPAInceptionv3(nn.Module):

    
    def __init__(self, pretrained=True, **kwargs):
        super(HPAInceptionv3, self).__init__(**kwargs)

        self.inception = models.inception_v3(pretrained=pretrained,transform_input=False,
                                             aux_logits=True)

        # change the first conv layer from 7,7,3,64 -> 7,7,4,64
        # weight of the last layer is taken as the average of the first and thrid layer
        self.conv0 = nn.Conv2d(4, 32, kernel_size=3, stride=2, bias=False)
        w = self.inception.Conv2d_1a_3x3.conv.weight
        self.conv0.weight = nn.Parameter(torch.cat((w, (w[:,:0,:,:]+w[:,1:,:,:])*0.5), dim=1))
        self.module_1a = nn.Sequential(
            self.conv0,
            self.inception.Conv2d_1a_3x3.bn,
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)
#        
        self.logit = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                # nn.Linear(1024, 512, bias=True),
                # nn.ReLU(inplace=True),
                # nn.BatchNorm1d(512),
                # nn.Dropout(0.5),
                # nn.Linear(512, 28, bias=True),
                nn.Linear(2048,28)
                )
    
    def forward(self, x):
        batch_size, C, H, W = x.shape
        #input sie: (batch_size, 3, 224, 224)
        # x = x[:,0:3,:,:]
        # x = x.clone()
        x[:, 0] = (x[:, 0] - 0.5) / 0.5
        x[:, 1] = (x[:, 1] - 0.5) / 0.5
        x[:, 2] = (x[:, 2] - 0.5) / 0.5
        x[:, 3] = (x[:, 3] - 0.5) / 0.5
        # x[:, 0, :, :] = (x[:, 0, :, :] - mean[0])# / std[0]
        # x[:, 1, :, :] = (x[:, 1, :, :] - mean[1])# / std[1]
        # x[:, 2, :, :] = (x[:, 2, :, :] - mean[2])# / std[2]
        # x = x[:,0:3,:,:]
        x = self.module_1a(x)                                      #; print('x', x.size())
        # x = self.inception.Conv2d_1a_3x3(x)  # 149 x 149 x 32
        x = self.inception.Conv2d_2a_3x3(x) # 147 x 147 x 32
        x = self.inception.Conv2d_2b_3x3(x) # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2) # 73 x 73 x 64
        x = self.inception.Conv2d_3b_1x1(x) # 73 x 73 x 80
        x = self.inception.Conv2d_4a_3x3(x) # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2) # 35 x 35 x 192
        x = self.inception.Mixed_5b(x) # 35 x 35 x 256
        x = self.inception.Mixed_5c(x) # 35 x 35 x 288
        x = self.inception.Mixed_5d(x) # 35 x 35 x 288
        x = self.inception.Mixed_6a(x) # 17 x 17 x 768
        x = self.inception.Mixed_6b(x) # 17 x 17 x 768
        x = self.inception.Mixed_6c(x) # 17 x 17 x 768
        x = self.inception.Mixed_6d(x) # 17 x 17 x 768
        x = self.inception.Mixed_6e(x) # 17 x 17 x 768

        x = self.inception.Mixed_7a(x) # 8 x 8 x 1280
        x = self.inception.Mixed_7b(x) # 8 x 8 x 2048
        x = self.inception.Mixed_7c(x) # 8 x 8 x 2048
        f = self.avgpool(x).view(batch_size, -1)+self.maxpool(x).view(batch_size, -1)                  #; print('f', f.size())
        # f = f.view(f.size(0), -1)                              #; print('f', f.size())
        logit = self.logit(f)                                  #; print('logit',logit.size())
        return logit        

if __name__ == '__main__':
    net = HPAInceptionv3()
    x = torch.rand(4, 4, 224, 224)
    net.forward(x)