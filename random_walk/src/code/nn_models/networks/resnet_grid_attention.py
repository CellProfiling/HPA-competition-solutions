import numpy as np
import math
import torch.nn as nn
from .utils import unetConv2, unetUp, conv2DBatchNormRelu, conv2DBatchNorm
import torch
import torch.nn.functional as F
from nn_models.layers.grid_attention_layer_parrallel import GridAttentionBlock2D_TORR as AttentionBlock2D
from nn_models.networks_other import init_weights
import torchvision

class resnet_grid_attention(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, in_channels=3, is_batchnorm=True, n_convs=None,
                 nonlocal_mode='concatenation', aggregation_mode='concat'):
        super(resnet_grid_attention, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes= n_classes
        self.aggregation_mode = aggregation_mode
        self.deep_supervised = True

        # # Resnet Pretrained Define
        self.resnet = torchvision.models.resnet18(pretrained=True)
        conv1_weight = self.resnet.conv1.weight.data
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.resnet.conv1.weight = torch.nn.Parameter(torch.cat((conv1_weight,conv1_weight[:,:1,:,:]),dim=1))
        filters = [64, 64, 128, 256, 512]
        # filters = [int(x / self.feature_scale) for x in filters]
            # # Feature Extraction
        self.conv1 = self.resnet.conv1
        self.conv2 = self.resnet.layer1
        self.conv3 = self.resnet.layer2
        self.conv4 = self.resnet.layer3
        self.conv5 = self.resnet.layer4


        # if n_convs is None:
        #     n_convs = [3, 3, 3, 2, 2]

        # filters = [64, 128, 256, 512]
        # filters = [int(x / self.feature_scale) for x in filters]

        ####################
        # # Feature Extraction
        # self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, n=n_convs[0])
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, n=n_convs[1])
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, n=n_convs[2])
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, n=n_convs[3])
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # self.conv5 = unetConv2(filters[3], filters[3], self.is_batchnorm, n=n_convs[4])

        ################
        # Attention Maps
        self.compatibility_score1 = AttentionBlock2D(in_channels=filters[2], gating_channels=filters[4],
                                                     inter_channels=filters[4], sub_sample_factor=(1,1),
                                                     mode=nonlocal_mode, use_W=False, use_phi=True,
                                                     use_theta=True, use_psi=True, nonlinearity1='relu')

        self.compatibility_score2 = AttentionBlock2D(in_channels=filters[3], gating_channels=filters[4],
                                                     inter_channels=filters[4], sub_sample_factor=(1,1),
                                                     mode=nonlocal_mode, use_W=False, use_phi=True,
                                                     use_theta=True, use_psi=True, nonlinearity1='relu')

        #########################
        # Aggreagation Strategies
        self.attention_filter_sizes = [filters[2], filters[3]]

        if aggregation_mode == 'concat':
            self.classifier = nn.Linear(filters[2]+filters[3]+filters[4], n_classes)
            self.aggregate = self.aggreagation_concat

        else:
            self.classifier1 = nn.Linear(filters[2], n_classes)
            self.classifier2 = nn.Linear(filters[3], n_classes)
            self.classifier3 = nn.Linear(filters[4], n_classes)
            self.classifiers = [self.classifier1, self.classifier2, self.classifier3]

            if aggregation_mode == 'mean':
                # self.aggregate = self.aggregation_sep
                self.aggregate = Aggregation_sep(self.classifiers)

            elif aggregation_mode == 'deep_sup':
                self.classifier = nn.Linear(filters[2] + filters[3] + filters[4], n_classes)
                # self.aggregate = self.aggregation_ds
                self.aggregate = Aggregation_ds(self.classifiers, self.classifier)

            elif aggregation_mode == 'ft':
                self.classifier = nn.Linear(n_classes*3, n_classes)
                # self.aggregate = self.aggregation_ft
                self.aggregate = Aggregation_ft(self.classifiers, self.classifier)
            else:
                raise NotImplementedError

        ####################
        # initialise weights
        # Freezing BatchNorm2D
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
                # m.eval()
                # # shutdown update in frozen mode
                # m.weight.requires_grad = False
                # m.bias.requires_grad = False


    def aggregation_sep(self, *attended_maps):
        return [ clf(att) for clf, att in zip(self.classifiers, attended_maps) ]

    def aggregation_ft(self, *attended_maps):
        # preds =  self.aggregation_sep(*attended_maps)
        preds = Aggregation_sep(self.classifiers)(*attended_maps)
        return self.classifier(torch.cat(preds, dim=1))

    def aggregation_ds(self, *attended_maps):
        preds_sep =  self.aggregation_sep(*attended_maps)
        pred = self.aggregation_concat(*attended_maps)
        return [pred] + preds_sep

    def aggregation_concat(self, *attended_maps):
        return self.classifier(torch.cat(attended_maps, dim=1))


    def forward(self, inputs):
        # # Feature Extraction
        # conv1    = self.conv1(inputs)
        # maxpool1 = self.maxpool1(conv1)

        # conv2    = self.conv2(maxpool1)
        # maxpool2 = self.maxpool2(conv2)

        # conv3    = self.conv3(maxpool2)
        # maxpool3 = self.maxpool3(conv3)

        # conv4    = self.conv4(maxpool3)
        # maxpool4 = self.maxpool4(conv4)

        # conv5    = self.conv5(maxpool4)

        # Feature Extraction
        conv1    = self.conv1(inputs)
        conv2    = self.conv2(conv1)
        conv3    = self.conv3(conv2)
        conv4    = self.conv4(conv3)
        conv5    = self.conv5(conv4)

        batch_size = inputs.shape[0]
        pooled     = F.adaptive_avg_pool2d(conv5, (1, 1)).view(batch_size, -1)

        # Attention Mechanism
        g_conv1, att1 = self.compatibility_score1(conv3, conv5)
        g_conv2, att2 = self.compatibility_score2(conv4, conv5)

        # flatten to get single feature vector
        fsizes = self.attention_filter_sizes
        g1 = torch.sum(g_conv1.view(batch_size, fsizes[0], -1), dim=-1)
        g2 = torch.sum(g_conv2.view(batch_size, fsizes[1], -1), dim=-1)

        return self.aggregate(g1, g2, pooled)


class Aggregation_sep(nn.Module):
    """docstring for Aggregation_sep"""
    def __init__(self, classifiers):
        super(Aggregation_sep, self).__init__()
        self.classifier1, self.classifier2, self.classifier3 = classifiers

    def forward(self, g1, g2, pooled):
        g1_cls = self.classifier1(g1)
        g2_cls = self.classifier2(g2)
        pooled_cls = self.classifier3(pooled)
        return g1_cls, g2_cls, pooled_cls

class Aggregation_ft(nn.Module):
    """docstring for Aggregation_ft"""
    def __init__(self, classifiers, classifier):
        super(Aggregation_ft, self).__init__()
        self.classifier1, self.classifier2, self.classifier3 = classifiers
        self.classifier = classifier
    def forward(self, g1, g2, pooled):
        g1_cls = self.classifier1(g1)
        g2_cls = self.classifier2(g2)
        pooled_cls = self.classifier3(pooled)
        logit = self.classifier(torch.cat([g1_cls, g2_cls, pooled_cls], dim=1))
        return logit

class Aggregation_ds(nn.Module):
    def __init__(self, classifiers, classifier):
        super(Aggregation_ds, self).__init__()
        self.classifier1, self.classifier2, self.classifier3 = classifiers
        self.classifier = classifier

    def forward(self, g1, g2, pooled):
        g1_cls = self.classifier1(g1)
        g2_cls = self.classifier2(g2)
        pooled_cls = self.classifier3(pooled)
        pred = self.classifier(torch.cat([g1,g2,pooled], dim=1))
        return pred, g1_cls, g2_cls, pooled_cls




