import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.backbone.inception_v3 import *

class ClassInceptionV3(nn.Module):

    def load_pretrain(self, pretrain_file):
        print('loading pretrain...%s' % pretrain_file)
        self.inception.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

    def __init__(self, num_classes=28,
                 in_channels=3,
                 pretrained_file=None,
                 dropout=False,):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout

        self.inception = inception_v3()
        self.load_pretrain(pretrained_file)
        if self.in_channels > 3:
            # https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
            w = self.inception.Conv2d_1a_3x3.conv.weight
            self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
            self.inception.Conv2d_1a_3x3.conv.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))

        self.Conv2d_1a_3x3 = self.inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = self.inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = self.inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = self.inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = self.inception.Conv2d_4a_3x3
        self.Mixed_5b = self.inception.Mixed_5b
        self.Mixed_5c = self.inception.Mixed_5c
        self.Mixed_5d = self.inception.Mixed_5d
        self.Mixed_6a = self.inception.Mixed_6a
        self.Mixed_6b = self.inception.Mixed_6b
        self.Mixed_6c = self.inception.Mixed_6c
        self.Mixed_6d = self.inception.Mixed_6d
        self.Mixed_6e = self.inception.Mixed_6e
        self.Mixed_7a = self.inception.Mixed_7a
        self.Mixed_7b = self.inception.Mixed_7b
        self.Mixed_7c = self.inception.Mixed_7c
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(2048, num_classes)

        # https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        if self.dropout:
            self.bn1 = nn.BatchNorm1d(2048 * 2)
            self.fc1 = nn.Linear(2048 * 2, 2048)
            self.bn2 = nn.BatchNorm1d(2048)
            self.relu = nn.ReLU(inplace=True)
        self.extract_feature = False

    def set_configs(self, extract_feature=False, **kwargs):
        self.extract_feature = extract_feature

    def forward(self, x):
        mean = [0.074598, 0.050630, 0.050891, 0.076287]#rgby
        std =  [0.122813, 0.085745, 0.129882, 0.119411]
        for i in range(self.in_channels):
            x[:,i,:,:] = (x[:,i,:,:] - mean[i]) / std[i]

        x = self.Conv2d_1a_3x3(x)#;print(x.size())
        x = self.Conv2d_2a_3x3(x)#;print(x.size())
        x = self.Conv2d_2b_3x3(x)#;print(x.size())
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x);#print(x.size())
        if self.dropout:
            x = torch.cat((nn.AdaptiveAvgPool2d(1)(x), nn.AdaptiveMaxPool2d(1)(x)), dim=1)
            x = x.view(x.size(0), -1)
            x = self.bn1(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.bn2(x)
            x = F.dropout(x, p=0.5, training=self.training)
        else:
            x = self.avgpool(x)
        x_feats = x.view(x.size(0), -1)
        x = self.logit(x_feats)
        if self.extract_feature:
            return x, x_feats
        else:
            return x

def class_inceptionv3_dropout(**kwargs):
    num_classes = kwargs['num_classes']
    in_channels = kwargs['in_channels']
    pretrained_file = kwargs['pretrained_file']
    model = ClassInceptionV3(num_classes=num_classes,
                             in_channels=in_channels, pretrained_file=pretrained_file, dropout=True)
    return model
