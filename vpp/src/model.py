#coding=utf-8
from pretrainedmodels.models import *
from torch import nn
import types
import sys
sys.path.append('../')
import torch.nn.functional as F

def get_net_inceptionv3(config):
    model = inceptionv3(pretrained="imagenet")
    model.aux_logits = False
    model.Conv2d_1a_3x3.conv = nn.Conv2d(config.channels, 32, bias=False, kernel_size=3, stride=2)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048, config.num_classes),
            )
    def logits(self, features):
        x = F.adaptive_avg_pool2d(features,1) # 1 x 1 x 2048
        x = x.view(x.size(0), -1) # 2048
        x = self.last_linear(x) # 1000 (num_classes)
        return x
    model.logits = types.MethodType(logits, model)

    return model
def get_net_inceptionv3_fc(config):
    model = inceptionv3(pretrained="imagenet")
    model.aux_logits = False
    model.Conv2d_1a_3x3.conv = nn.Conv2d(config.channels, 32, bias=False, kernel_size=3, stride=2)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048,  128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, config.num_classes),
            )

    def logits(self, features):
        x = F.adaptive_avg_pool2d(features,1) # 1 x 1 x 2048
        x = x.view(x.size(0), -1) # 2048
        x = self.last_linear(x) # 1000 (num_classes)
        return x
    model.logits = types.MethodType(logits, model)

    if config.run_type=='feature':
        def forward(self, input):
            x = self.features(input)
            x = F.adaptive_avg_pool2d(x, 1)  # 1 x 1 x 2048
            x = x.view(x.size(0), -1)  # 2048
            x_feature = self.last_linear[:3](x)
            x_class = self.last_linear[3:](x_feature)
            return x_class,x_feature

        model.forward = types.MethodType(forward, model)

    return model

def get_net_inceptionv4(config):

    model = inceptionv4(pretrained="imagenet")
    model.features[0].conv = nn.Conv2d(config.channels, 32,bias=False, kernel_size=3, stride=2)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1536),
                nn.Dropout(0.5),
                nn.Linear(1536, config.num_classes),
            )
    return model

def get_net_inceptionv4_fc(config):

    model = inceptionv4(pretrained="imagenet")
    model.features[0].conv = nn.Conv2d(config.channels, 32,bias=False, kernel_size=3, stride=2)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1536),
                nn.Dropout(0.5),
                nn.Linear(1536, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, config.num_classes),
            )

    if config.run_type=='feature':
        def forward(self, input):
            x = self.features(input)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)  # 2048
            x_feature = self.last_linear[:3](x)
            x_class = self.last_linear[3:](x_feature)
            return x_class,x_feature

        model.forward = types.MethodType(forward, model)
    return model


def get_net_xception(config):

    model = xception(pretrained="imagenet")
    model.conv1 = nn.Conv2d(config.channels, 32, 3,2, 0, bias=False)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048, config.num_classes),
            )
    return model
def get_net_xception_fc(config):

    model = xception(pretrained="imagenet")
    if config.channels!=3:
        model.conv1 = nn.Conv2d(config.channels, 32, 3,2, 0, bias=False)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048,  128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, config.num_classes),
            )

    if config.run_type=='feature':
        def forward(self, input):
            x = self.features(input)
            x = self.relu(x)

            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)

            x_feature = self.last_linear[:3](x)
            x_class = self.last_linear[3:](x_feature)
            return x_class,x_feature

        model.forward = types.MethodType(forward, model)

    return model


def get_net(config):
    print('****************** Use {} ******************'.format(config.model_name))
    if  config.model_name == 'xception':
        return get_net_xception(config)
    elif config.model_name == 'xception_fc':
        return get_net_xception_fc(config)
    elif config.model_name == 'inceptionv3':
        return get_net_inceptionv3(config)
    elif config.model_name == 'inceptionv3_fc':
        return get_net_inceptionv3_fc(config)
    elif config.model_name == 'inceptionv4':
        return get_net_inceptionv4(config)
    elif config.model_name == 'inceptionv4_fc':
        return get_net_inceptionv4_fc(config)
    else:
        print('Error model {} not found!'.format(config.model_name))
        sys.exit('0')

