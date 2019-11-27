import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.backbone.resnet import *

## net  ######################################################################
class ResnetClass(nn.Module):

    def load_pretrain(self, pretrain_file):
        print('loading pretrain...%s' % pretrain_file)
        self.resnet.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

    def __init__(self,
                 feature_net='resnet34',
                 num_classes=28,
                 in_channels=3,
                 pretrained_file=None,
                 dropout=False,
                 ):
        super().__init__()
        self.dropout = dropout

        if feature_net == 'resnet18':
            self.resnet = resnet18()
            self.EX = 1
        elif feature_net=='resnet34':
            self.resnet = resnet34()
            self.EX=1
        elif feature_net=='resnet50':
            self.resnet = resnet50()
            self.EX = 4
        elif feature_net=='resnet101':
            self.resnet = resnet101()
            self.EX = 4
        elif feature_net=='resnet152':
            self.resnet = resnet152()
            self.EX = 4

        self.load_pretrain(pretrained_file)
        self.in_channels = in_channels
        if self.in_channels > 3:
            # https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
            w = self.resnet.conv1.weight
            self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3, 3), bias=False)
            self.resnet.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:,:1,:,:]),dim=1))

        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(512 * self.EX, num_classes)

        # https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        if self.dropout:
            self.bn1 = nn.BatchNorm1d(1024 * self.EX)
            self.fc1 = nn.Linear(1024 * self.EX, 512 * self.EX)
            self.bn2 = nn.BatchNorm1d(512 * self.EX)
            self.relu = nn.ReLU(inplace=True)
        self.extract_feature = False

    def set_configs(self, extract_feature=False, **kwargs):
        self.extract_feature = extract_feature

    def forward(self, x):
        mean = [0.074598, 0.050630, 0.050891, 0.076287]#rgby
        std =  [0.122813, 0.085745, 0.129882, 0.119411]
        for i in range(self.in_channels):
            x[:,i,:,:] = (x[:,i,:,:] - mean[i]) / std[i]

        x = self.encoder1(x)
        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        if self.dropout:
            x = torch.cat((nn.AdaptiveAvgPool2d(1)(e5), nn.AdaptiveMaxPool2d(1)(e5)), dim=1)
            x = x.view(x.size(0), -1)
            x = self.bn1(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.bn2(x)
            x = F.dropout(x, p=0.5, training=self.training)
        else:
            x = self.avgpool(e5)
        feature = x.view(x.size(0), -1)
        x = self.logit(feature)

        if self.extract_feature:
            return x, feature
        else:
            return x

def class_resnet34_dropout(**kwargs):
    num_classes = kwargs['num_classes']
    in_channels = kwargs['in_channels']
    pretrained_file = kwargs['pretrained_file']
    model = ResnetClass(feature_net='resnet34', num_classes=num_classes,
                        in_channels=in_channels, pretrained_file=pretrained_file, dropout=True)
    return model

def class_resnet18_dropout(**kwargs):
    num_classes = kwargs['num_classes']
    in_channels = kwargs['in_channels']
    pretrained_file = kwargs['pretrained_file']
    model = ResnetClass(feature_net='resnet18', num_classes=num_classes,
                        in_channels=in_channels, pretrained_file=pretrained_file, dropout=True)
    return model
