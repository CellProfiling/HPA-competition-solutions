from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels
import pretrainedmodels.models.senet
import config
from pytorchcv.model_provider import get_model as ptcv_get_model


class ClassificationModel(nn.Module):
    def __init__(self, base_model, nb_features, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features

        self.fc = nn.Linear(nb_features * 2, config.NB_CATEGORIES)

        self.l1_4 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.layer0[0] = self.l1_4

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        features = self.base_model.features(inputs)

        avg_pool = F.avg_pool2d(features, features.shape[2:])
        max_pool = F.max_pool2d(features, features.shape[2:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        out = avg_max_pool.view(avg_max_pool.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = self.fc(out)
        # out = self.output_act(out)

        return out


class ClassificationModelResnet(nn.Module):
    def __init__(self, base_model, nb_features, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features

        self.fc = nn.Linear(nb_features * 2, config.NB_CATEGORIES)

        self.l1_4 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.conv1 = self.l1_4

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        x = self.l1_4(inputs)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        features = self.base_model.layer4(x)

        avg_pool = F.avg_pool2d(features, features.shape[2:])
        max_pool = F.max_pool2d(features, features.shape[2:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        out = avg_max_pool.view(avg_max_pool.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = self.fc(out)
        # out = self.output_act(out)

        return out


class ClassificationModelBnInception(nn.Module):
    def __init__(self, base_model, nb_features, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features

        self.fc = nn.Linear(nb_features * 2, config.NB_CATEGORIES)

        self.l1_4 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.base_model.conv1_7x7_s2 = self.l1_4

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        features = self.base_model.features(inputs)

        avg_pool = F.avg_pool2d(features, features.shape[2:])
        max_pool = F.max_pool2d(features, features.shape[2:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        out = avg_max_pool.view(avg_max_pool.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = self.fc(out)
        # out = self.output_act(out)

        return out


class ClassificationModelBnInceptionWGAP(nn.Module):
    def __init__(self,
                 base_model,
                 gwap_channels=(100, 200, 300, 400),
                 fc_layers=(888, 256),
                 scale=3,
                 dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model

        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = sum(gwap_channels)

        # self.gwap = {}
        # for i, ch in enumerate(gwap_channels):
        #     gwap = GWAP(ch, scale=scale)
        #     self.gwap[i] = gwap
        #     self.add_module(f'gwap{i}', gwap)

        self.wgap0 = GWAP(gwap_channels[0], scale=scale)
        self.wgap1 = GWAP(gwap_channels[1], scale=scale)
        self.wgap2 = GWAP(gwap_channels[2], scale=scale)
        self.wgap3 = GWAP(gwap_channels[3], scale=scale)

        self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])
        self.fc2 = nn.Linear(fc_layers[1], config.NB_CATEGORIES)

        self.l1_4 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.base_model.conv1_7x7_s2 = self.l1_4

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        conv1_7x7_s2_out = self.base_model.conv1_7x7_s2(inputs)
        conv1_7x7_s2_bn_out = self.base_model.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.base_model.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.base_model.pool1_3x3_s2(conv1_relu_7x7_out)
        conv2_3x3_reduce_out = self.base_model.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.base_model.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.base_model.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.base_model.conv2_3x3(conv2_relu_3x3_reduce_out)
        conv2_3x3_bn_out = self.base_model.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.base_model.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.base_model.pool2_3x3_s2(conv2_relu_3x3_out)
        inception_3a_1x1_out = self.base_model.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.base_model.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.base_model.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.base_model.inception_3a_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.base_model.inception_3a_3x3_reduce_bn(inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = self.base_model.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.base_model.inception_3a_3x3(inception_3a_relu_3x3_reduce_out)
        inception_3a_3x3_bn_out = self.base_model.inception_3a_3x3_bn(inception_3a_3x3_out)
        inception_3a_relu_3x3_out = self.base_model.inception_3a_relu_3x3(inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = self.base_model.inception_3a_double_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_double_3x3_reduce_bn_out = self.base_model.inception_3a_double_3x3_reduce_bn(
            inception_3a_double_3x3_reduce_out)
        inception_3a_relu_double_3x3_reduce_out = self.base_model.inception_3a_relu_double_3x3_reduce(
            inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_out = self.base_model.inception_3a_double_3x3_1(inception_3a_relu_double_3x3_reduce_out)
        inception_3a_double_3x3_1_bn_out = self.base_model.inception_3a_double_3x3_1_bn(inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = self.base_model.inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_out = self.base_model.inception_3a_double_3x3_2(inception_3a_relu_double_3x3_1_out)
        inception_3a_double_3x3_2_bn_out = self.base_model.inception_3a_double_3x3_2_bn(inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = self.base_model.inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out)
        inception_3a_pool_out = self.base_model.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.base_model.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.base_model.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.base_model.inception_3a_relu_pool_proj(inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat(
            [inception_3a_relu_1x1_out, inception_3a_relu_3x3_out, inception_3a_relu_double_3x3_2_out,
             inception_3a_relu_pool_proj_out], 1)
        inception_3b_1x1_out = self.base_model.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.base_model.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.base_model.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.base_model.inception_3b_3x3_reduce(inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.base_model.inception_3b_3x3_reduce_bn(inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = self.base_model.inception_3b_relu_3x3_reduce(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.base_model.inception_3b_3x3(inception_3b_relu_3x3_reduce_out)
        inception_3b_3x3_bn_out = self.base_model.inception_3b_3x3_bn(inception_3b_3x3_out)
        inception_3b_relu_3x3_out = self.base_model.inception_3b_relu_3x3(inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = self.base_model.inception_3b_double_3x3_reduce(inception_3a_output_out)
        inception_3b_double_3x3_reduce_bn_out = self.base_model.inception_3b_double_3x3_reduce_bn(
            inception_3b_double_3x3_reduce_out)
        inception_3b_relu_double_3x3_reduce_out = self.base_model.inception_3b_relu_double_3x3_reduce(
            inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_out = self.base_model.inception_3b_double_3x3_1(inception_3b_relu_double_3x3_reduce_out)
        inception_3b_double_3x3_1_bn_out = self.base_model.inception_3b_double_3x3_1_bn(inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = self.base_model.inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_out = self.base_model.inception_3b_double_3x3_2(inception_3b_relu_double_3x3_1_out)
        inception_3b_double_3x3_2_bn_out = self.base_model.inception_3b_double_3x3_2_bn(inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = self.base_model.inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out)
        inception_3b_pool_out = self.base_model.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.base_model.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.base_model.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.base_model.inception_3b_relu_pool_proj(inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat(
            [inception_3b_relu_1x1_out, inception_3b_relu_3x3_out, inception_3b_relu_double_3x3_2_out,
             inception_3b_relu_pool_proj_out], 1)
        inception_3c_3x3_reduce_out = self.base_model.inception_3c_3x3_reduce(inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.base_model.inception_3c_3x3_reduce_bn(inception_3c_3x3_reduce_out)
        inception_3c_relu_3x3_reduce_out = self.base_model.inception_3c_relu_3x3_reduce(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_out = self.base_model.inception_3c_3x3(inception_3c_relu_3x3_reduce_out)
        inception_3c_3x3_bn_out = self.base_model.inception_3c_3x3_bn(inception_3c_3x3_out)
        inception_3c_relu_3x3_out = self.base_model.inception_3c_relu_3x3(inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = self.base_model.inception_3c_double_3x3_reduce(inception_3b_output_out)
        inception_3c_double_3x3_reduce_bn_out = self.base_model.inception_3c_double_3x3_reduce_bn(
            inception_3c_double_3x3_reduce_out)
        inception_3c_relu_double_3x3_reduce_out = self.base_model.inception_3c_relu_double_3x3_reduce(
            inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_out = self.base_model.inception_3c_double_3x3_1(inception_3c_relu_double_3x3_reduce_out)
        inception_3c_double_3x3_1_bn_out = self.base_model.inception_3c_double_3x3_1_bn(inception_3c_double_3x3_1_out)
        inception_3c_relu_double_3x3_1_out = self.base_model.inception_3c_relu_double_3x3_1(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_out = self.base_model.inception_3c_double_3x3_2(inception_3c_relu_double_3x3_1_out)
        inception_3c_double_3x3_2_bn_out = self.base_model.inception_3c_double_3x3_2_bn(inception_3c_double_3x3_2_out)
        inception_3c_relu_double_3x3_2_out = self.base_model.inception_3c_relu_double_3x3_2(inception_3c_double_3x3_2_bn_out)
        inception_3c_pool_out = self.base_model.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat(
            [inception_3c_relu_3x3_out, inception_3c_relu_double_3x3_2_out, inception_3c_pool_out], 1)
        inception_4a_1x1_out = self.base_model.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.base_model.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = self.base_model.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.base_model.inception_4a_3x3_reduce(inception_3c_output_out)
        inception_4a_3x3_reduce_bn_out = self.base_model.inception_4a_3x3_reduce_bn(inception_4a_3x3_reduce_out)
        inception_4a_relu_3x3_reduce_out = self.base_model.inception_4a_relu_3x3_reduce(inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_out = self.base_model.inception_4a_3x3(inception_4a_relu_3x3_reduce_out)
        inception_4a_3x3_bn_out = self.base_model.inception_4a_3x3_bn(inception_4a_3x3_out)
        inception_4a_relu_3x3_out = self.base_model.inception_4a_relu_3x3(inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = self.base_model.inception_4a_double_3x3_reduce(inception_3c_output_out)
        inception_4a_double_3x3_reduce_bn_out = self.base_model.inception_4a_double_3x3_reduce_bn(
            inception_4a_double_3x3_reduce_out)
        inception_4a_relu_double_3x3_reduce_out = self.base_model.inception_4a_relu_double_3x3_reduce(
            inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_out = self.base_model.inception_4a_double_3x3_1(inception_4a_relu_double_3x3_reduce_out)
        inception_4a_double_3x3_1_bn_out = self.base_model.inception_4a_double_3x3_1_bn(inception_4a_double_3x3_1_out)
        inception_4a_relu_double_3x3_1_out = self.base_model.inception_4a_relu_double_3x3_1(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_out = self.base_model.inception_4a_double_3x3_2(inception_4a_relu_double_3x3_1_out)
        inception_4a_double_3x3_2_bn_out = self.base_model.inception_4a_double_3x3_2_bn(inception_4a_double_3x3_2_out)
        inception_4a_relu_double_3x3_2_out = self.base_model.inception_4a_relu_double_3x3_2(inception_4a_double_3x3_2_bn_out)
        inception_4a_pool_out = self.base_model.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.base_model.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.base_model.inception_4a_pool_proj_bn(inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.base_model.inception_4a_relu_pool_proj(inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat(
            [inception_4a_relu_1x1_out, inception_4a_relu_3x3_out, inception_4a_relu_double_3x3_2_out,
             inception_4a_relu_pool_proj_out], 1)
        inception_4b_1x1_out = self.base_model.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.base_model.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = self.base_model.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.base_model.inception_4b_3x3_reduce(inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.base_model.inception_4b_3x3_reduce_bn(inception_4b_3x3_reduce_out)
        inception_4b_relu_3x3_reduce_out = self.base_model.inception_4b_relu_3x3_reduce(inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_out = self.base_model.inception_4b_3x3(inception_4b_relu_3x3_reduce_out)
        inception_4b_3x3_bn_out = self.base_model.inception_4b_3x3_bn(inception_4b_3x3_out)
        inception_4b_relu_3x3_out = self.base_model.inception_4b_relu_3x3(inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = self.base_model.inception_4b_double_3x3_reduce(inception_4a_output_out)
        inception_4b_double_3x3_reduce_bn_out = self.base_model.inception_4b_double_3x3_reduce_bn(
            inception_4b_double_3x3_reduce_out)
        inception_4b_relu_double_3x3_reduce_out = self.base_model.inception_4b_relu_double_3x3_reduce(
            inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_out = self.base_model.inception_4b_double_3x3_1(inception_4b_relu_double_3x3_reduce_out)
        inception_4b_double_3x3_1_bn_out = self.base_model.inception_4b_double_3x3_1_bn(inception_4b_double_3x3_1_out)
        inception_4b_relu_double_3x3_1_out = self.base_model.inception_4b_relu_double_3x3_1(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_out = self.base_model.inception_4b_double_3x3_2(inception_4b_relu_double_3x3_1_out)
        inception_4b_double_3x3_2_bn_out = self.base_model.inception_4b_double_3x3_2_bn(inception_4b_double_3x3_2_out)
        inception_4b_relu_double_3x3_2_out = self.base_model.inception_4b_relu_double_3x3_2(inception_4b_double_3x3_2_bn_out)
        inception_4b_pool_out = self.base_model.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.base_model.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.base_model.inception_4b_pool_proj_bn(inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.base_model.inception_4b_relu_pool_proj(inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat(
            [inception_4b_relu_1x1_out, inception_4b_relu_3x3_out, inception_4b_relu_double_3x3_2_out,
             inception_4b_relu_pool_proj_out], 1)
        inception_4c_1x1_out = self.base_model.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.base_model.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = self.base_model.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.base_model.inception_4c_3x3_reduce(inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.base_model.inception_4c_3x3_reduce_bn(inception_4c_3x3_reduce_out)
        inception_4c_relu_3x3_reduce_out = self.base_model.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_out = self.base_model.inception_4c_3x3(inception_4c_relu_3x3_reduce_out)
        inception_4c_3x3_bn_out = self.base_model.inception_4c_3x3_bn(inception_4c_3x3_out)
        inception_4c_relu_3x3_out = self.base_model.inception_4c_relu_3x3(inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = self.base_model.inception_4c_double_3x3_reduce(inception_4b_output_out)
        inception_4c_double_3x3_reduce_bn_out = self.base_model.inception_4c_double_3x3_reduce_bn(
            inception_4c_double_3x3_reduce_out)
        inception_4c_relu_double_3x3_reduce_out = self.base_model.inception_4c_relu_double_3x3_reduce(
            inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_out = self.base_model.inception_4c_double_3x3_1(inception_4c_relu_double_3x3_reduce_out)
        inception_4c_double_3x3_1_bn_out = self.base_model.inception_4c_double_3x3_1_bn(inception_4c_double_3x3_1_out)
        inception_4c_relu_double_3x3_1_out = self.base_model.inception_4c_relu_double_3x3_1(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_out = self.base_model.inception_4c_double_3x3_2(inception_4c_relu_double_3x3_1_out)
        inception_4c_double_3x3_2_bn_out = self.base_model.inception_4c_double_3x3_2_bn(inception_4c_double_3x3_2_out)
        inception_4c_relu_double_3x3_2_out = self.base_model.inception_4c_relu_double_3x3_2(inception_4c_double_3x3_2_bn_out)
        inception_4c_pool_out = self.base_model.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.base_model.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.base_model.inception_4c_pool_proj_bn(inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.base_model.inception_4c_relu_pool_proj(inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat(
            [inception_4c_relu_1x1_out, inception_4c_relu_3x3_out, inception_4c_relu_double_3x3_2_out,
             inception_4c_relu_pool_proj_out], 1)
        inception_4d_1x1_out = self.base_model.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.base_model.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = self.base_model.inception_4d_relu_1x1(inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.base_model.inception_4d_3x3_reduce(inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.base_model.inception_4d_3x3_reduce_bn(inception_4d_3x3_reduce_out)
        inception_4d_relu_3x3_reduce_out = self.base_model.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_out = self.base_model.inception_4d_3x3(inception_4d_relu_3x3_reduce_out)
        inception_4d_3x3_bn_out = self.base_model.inception_4d_3x3_bn(inception_4d_3x3_out)
        inception_4d_relu_3x3_out = self.base_model.inception_4d_relu_3x3(inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = self.base_model.inception_4d_double_3x3_reduce(inception_4c_output_out)
        inception_4d_double_3x3_reduce_bn_out = self.base_model.inception_4d_double_3x3_reduce_bn(
            inception_4d_double_3x3_reduce_out)
        inception_4d_relu_double_3x3_reduce_out = self.base_model.inception_4d_relu_double_3x3_reduce(
            inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_out = self.base_model.inception_4d_double_3x3_1(inception_4d_relu_double_3x3_reduce_out)
        inception_4d_double_3x3_1_bn_out = self.base_model.inception_4d_double_3x3_1_bn(inception_4d_double_3x3_1_out)
        inception_4d_relu_double_3x3_1_out = self.base_model.inception_4d_relu_double_3x3_1(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_out = self.base_model.inception_4d_double_3x3_2(inception_4d_relu_double_3x3_1_out)
        inception_4d_double_3x3_2_bn_out = self.base_model.inception_4d_double_3x3_2_bn(inception_4d_double_3x3_2_out)
        inception_4d_relu_double_3x3_2_out = self.base_model.inception_4d_relu_double_3x3_2(inception_4d_double_3x3_2_bn_out)
        inception_4d_pool_out = self.base_model.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.base_model.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.base_model.inception_4d_pool_proj_bn(inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.base_model.inception_4d_relu_pool_proj(inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat(
            [inception_4d_relu_1x1_out, inception_4d_relu_3x3_out, inception_4d_relu_double_3x3_2_out,
             inception_4d_relu_pool_proj_out], 1)
        inception_4e_3x3_reduce_out = self.base_model.inception_4e_3x3_reduce(inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.base_model.inception_4e_3x3_reduce_bn(inception_4e_3x3_reduce_out)
        inception_4e_relu_3x3_reduce_out = self.base_model.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_out = self.base_model.inception_4e_3x3(inception_4e_relu_3x3_reduce_out)
        inception_4e_3x3_bn_out = self.base_model.inception_4e_3x3_bn(inception_4e_3x3_out)
        inception_4e_relu_3x3_out = self.base_model.inception_4e_relu_3x3(inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = self.base_model.inception_4e_double_3x3_reduce(inception_4d_output_out)
        inception_4e_double_3x3_reduce_bn_out = self.base_model.inception_4e_double_3x3_reduce_bn(
            inception_4e_double_3x3_reduce_out)
        inception_4e_relu_double_3x3_reduce_out = self.base_model.inception_4e_relu_double_3x3_reduce(
            inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_out = self.base_model.inception_4e_double_3x3_1(inception_4e_relu_double_3x3_reduce_out)
        inception_4e_double_3x3_1_bn_out = self.base_model.inception_4e_double_3x3_1_bn(inception_4e_double_3x3_1_out)
        inception_4e_relu_double_3x3_1_out = self.base_model.inception_4e_relu_double_3x3_1(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_out = self.base_model.inception_4e_double_3x3_2(inception_4e_relu_double_3x3_1_out)
        inception_4e_double_3x3_2_bn_out = self.base_model.inception_4e_double_3x3_2_bn(inception_4e_double_3x3_2_out)
        inception_4e_relu_double_3x3_2_out = self.base_model.inception_4e_relu_double_3x3_2(inception_4e_double_3x3_2_bn_out)
        inception_4e_pool_out = self.base_model.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat(
            [inception_4e_relu_3x3_out, inception_4e_relu_double_3x3_2_out, inception_4e_pool_out], 1)
        inception_5a_1x1_out = self.base_model.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.base_model.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = self.base_model.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.base_model.inception_5a_3x3_reduce(inception_4e_output_out)
        inception_5a_3x3_reduce_bn_out = self.base_model.inception_5a_3x3_reduce_bn(inception_5a_3x3_reduce_out)
        inception_5a_relu_3x3_reduce_out = self.base_model.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_out = self.base_model.inception_5a_3x3(inception_5a_relu_3x3_reduce_out)
        inception_5a_3x3_bn_out = self.base_model.inception_5a_3x3_bn(inception_5a_3x3_out)
        inception_5a_relu_3x3_out = self.base_model.inception_5a_relu_3x3(inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = self.base_model.inception_5a_double_3x3_reduce(inception_4e_output_out)
        inception_5a_double_3x3_reduce_bn_out = self.base_model.inception_5a_double_3x3_reduce_bn(
            inception_5a_double_3x3_reduce_out)
        inception_5a_relu_double_3x3_reduce_out = self.base_model.inception_5a_relu_double_3x3_reduce(
            inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_out = self.base_model.inception_5a_double_3x3_1(inception_5a_relu_double_3x3_reduce_out)
        inception_5a_double_3x3_1_bn_out = self.base_model.inception_5a_double_3x3_1_bn(inception_5a_double_3x3_1_out)
        inception_5a_relu_double_3x3_1_out = self.base_model.inception_5a_relu_double_3x3_1(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_out = self.base_model.inception_5a_double_3x3_2(inception_5a_relu_double_3x3_1_out)
        inception_5a_double_3x3_2_bn_out = self.base_model.inception_5a_double_3x3_2_bn(inception_5a_double_3x3_2_out)
        inception_5a_relu_double_3x3_2_out = self.base_model.inception_5a_relu_double_3x3_2(inception_5a_double_3x3_2_bn_out)
        inception_5a_pool_out = self.base_model.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.base_model.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.base_model.inception_5a_pool_proj_bn(inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.base_model.inception_5a_relu_pool_proj(inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat(
            [inception_5a_relu_1x1_out, inception_5a_relu_3x3_out, inception_5a_relu_double_3x3_2_out,
             inception_5a_relu_pool_proj_out], 1)
        inception_5b_1x1_out = self.base_model.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.base_model.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = self.base_model.inception_5b_relu_1x1(inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.base_model.inception_5b_3x3_reduce(inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.base_model.inception_5b_3x3_reduce_bn(inception_5b_3x3_reduce_out)
        inception_5b_relu_3x3_reduce_out = self.base_model.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_out = self.base_model.inception_5b_3x3(inception_5b_relu_3x3_reduce_out)
        inception_5b_3x3_bn_out = self.base_model.inception_5b_3x3_bn(inception_5b_3x3_out)
        inception_5b_relu_3x3_out = self.base_model.inception_5b_relu_3x3(inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = self.base_model.inception_5b_double_3x3_reduce(inception_5a_output_out)
        inception_5b_double_3x3_reduce_bn_out = self.base_model.inception_5b_double_3x3_reduce_bn(
            inception_5b_double_3x3_reduce_out)
        inception_5b_relu_double_3x3_reduce_out = self.base_model.inception_5b_relu_double_3x3_reduce(
            inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_out = self.base_model.inception_5b_double_3x3_1(inception_5b_relu_double_3x3_reduce_out)
        inception_5b_double_3x3_1_bn_out = self.base_model.inception_5b_double_3x3_1_bn(inception_5b_double_3x3_1_out)
        inception_5b_relu_double_3x3_1_out = self.base_model.inception_5b_relu_double_3x3_1(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_out = self.base_model.inception_5b_double_3x3_2(inception_5b_relu_double_3x3_1_out)
        inception_5b_double_3x3_2_bn_out = self.base_model.inception_5b_double_3x3_2_bn(inception_5b_double_3x3_2_out)
        inception_5b_relu_double_3x3_2_out = self.base_model.inception_5b_relu_double_3x3_2(inception_5b_double_3x3_2_bn_out)
        inception_5b_pool_out = self.base_model.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.base_model.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.base_model.inception_5b_pool_proj_bn(inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.base_model.inception_5b_relu_pool_proj(inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat(
            [inception_5b_relu_1x1_out, inception_5b_relu_3x3_out, inception_5b_relu_double_3x3_2_out,
             inception_5b_relu_pool_proj_out], 1)

        outputs = [
            self.wgap0(pool2_3x3_s2_out),
            self.wgap1(inception_3c_output_out),
            self.wgap2(inception_4e_output_out),
            self.wgap3(inception_5b_output_out),
        ]

        out = torch.cat(outputs, 1)
        out = out.view(out.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out


class ClassificationModelResnetWGAP(nn.Module):
    def __init__(self, base_model,
                 dropout=0.5,
                 gwap_channels=(64, 128, 256, 512),
                 scale=1,
                 fc_layers=(888, 256),
                 extra_gwap_layer=None,
                 trim_layers = 0
                 ):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.trim_layers = trim_layers

        self.gwap0 = GWAP(gwap_channels[0], scale=scale)
        self.gwap1 = GWAP(gwap_channels[1], scale=scale)
        self.gwap2 = GWAP(gwap_channels[2], scale=scale)
        self.gwap3 = GWAP(gwap_channels[3], scale=scale)

        self.fc_out_0 = nn.Linear(fc_layers[0], fc_layers[1])
        self.fc_out_1 = nn.Linear(fc_layers[1], config.NB_CATEGORIES)

        # self.gwap = []
        # for i, ch in enumerate(gwap_channels):
        #     gwap = GWAP(ch, scale=scale, extra_layer=extra_gwap_layer)
        #     self.gwap.append(gwap)
        #     self.add_module(f'gwap{i}', gwap)
        #
        # self.fc_layers = []
        # fc_layers = list(fc_layers) + [config.NB_CATEGORIES]
        # for i in range(len(fc_layers)-1):
        #     fc = nn.Linear(fc_layers[i], fc_layers[i+1])
        #     self.fc_layers.append(fc)
        #     self.add_module(f'fc_out_{i}', fc)

        self.l1_4 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.conv1 = self.l1_4

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        outputs = []

        x = self.l1_4(inputs)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        outputs.append(self.gwap0(x))
        x = self.base_model.layer2(x)
        outputs.append(self.gwap1(x))

        if self.trim_layers < 2:
            x = self.base_model.layer3(x)
            outputs.append(self.gwap2(x))

        if self.trim_layers < 1:
            x = self.base_model.layer4(x)
            outputs.append(self.gwap3(x))

        out = torch.cat(outputs, 1)
        out = out.view(out.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = self.fc_out_0(out)
        out = F.relu(out)
        out = self.fc_out_1(out)
        return out


class ClassificationModelInceptionResnetV2(nn.Module):
    def __init__(self, base_model, nb_features, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features

        self.fc = nn.Linear(nb_features * 2, config.NB_CATEGORIES)

        from pretrainedmodels.models.inceptionresnetv2 import BasicConv2d

        self.l1_4 = BasicConv2d(4, 32, kernel_size=3, stride=2)
        self.base_model.conv2d_1a = self.l1_4

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        features = self.base_model.features(inputs)

        avg_pool = F.avg_pool2d(features, features.shape[2:])
        max_pool = F.max_pool2d(features, features.shape[2:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        out = avg_max_pool.view(avg_max_pool.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = self.fc(out)
        # out = self.output_act(out)

        return out


class ClassificationModelDPN(nn.Module):
    def __init__(self, base_model, nb_features, num_init_features, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features

        self.fc = nn.Linear(nb_features * 2, config.NB_CATEGORIES)
        self.l1_4 = nn.Conv2d(4, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        # self.base_model.features['conv1_1']['conv'] = self.l1_4
        self.base_model.features[0].conv = self.l1_4

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        features = self.base_model.features(inputs)

        avg_pool = F.avg_pool2d(features, features.shape[2:])
        max_pool = F.max_pool2d(features, features.shape[2:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        out = avg_max_pool.view(avg_max_pool.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = self.fc(out)
        # out = self.output_act(out)

        return out


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class ClassificationModelResNextAttn(nn.Module):
    def __init__(self, base_model, nb_features, dropout=0.5, attn_levels=(2, 3, 4)):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.attn_levels = attn_levels

        self.fc = nn.Linear(nb_features[0] * 2, config.NB_CATEGORIES)

        self.l1_4 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.layer0[0] = self.l1_4

        self.attn2 = Self_Attn(nb_features[2])
        self.attn3 = Self_Attn(nb_features[1])
        self.attn4 = Self_Attn(nb_features[0])

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        # features = self.base_model.features(inputs)

        x = self.base_model.layer0(inputs)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        if 2 in self.attn_levels:
            x = self.attn2(x)
        x = self.base_model.layer3(x)
        if 3 in self.attn_levels:
            x = self.attn3(x)
        x = self.base_model.layer4(x)
        if 4 in self.attn_levels:
            x = self.attn4(x)
        features = x

        avg_pool = F.avg_pool2d(features, features.shape[2:])
        max_pool = F.max_pool2d(features, features.shape[2:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        out = avg_max_pool.view(avg_max_pool.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = self.fc(out)
        # out = self.output_act(out)

        return out


class GWAP(nn.Module):
    def __init__(self, channels, scale=1, extra_layer=None):
        super().__init__()
        self.scale = scale
        if extra_layer is not None:
            self.w1 = nn.Sequential(
                nn.Conv2d(channels, extra_layer, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(extra_layer, 1, kernel_size=1, bias=True)
            )
        else:
            self.w1 = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, inputs):
        inputs = inputs[:, :, 2:-2, 2:-2]  # discard borders

        x = self.w1(inputs)
        m = torch.exp(self.scale*torch.sigmoid(x))
        a = m / torch.sum(m, dim=(2, 3), keepdim=True)

        x = a * inputs
        gwap = torch.sum(x, dim=(2, 3))
        return gwap


class ClassificationModelResNextGap(nn.Module):
    def __init__(self, base_model, nb_features, dropout=0.5, gap_levels=(1, 2, 3)):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.gap_levels = gap_levels

        self.fc1 = nn.Linear(nb_features, 256)
        self.fc2 = nn.Linear(256, config.NB_CATEGORIES)

        self.l1_4 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.layer0[0] = self.l1_4

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        outputs = []

        def gap(x):
            x = F.avg_pool2d(x, x.shape[2:])
            return x

        x = self.base_model.layer0(inputs)
        x = self.base_model.layer1(x)
        if 1 in self.gap_levels:
            outputs.append(gap(x))
        x = self.base_model.layer2(x)
        if 2 in self.gap_levels:
            outputs.append(gap(x))
        x = self.base_model.layer3(x)
        if 3 in self.gap_levels:
            outputs.append(gap(x))
        x = self.base_model.layer4(x)
        features = x

        avg_pool = F.avg_pool2d(features, features.shape[2:])
        max_pool = F.max_pool2d(features, features.shape[2:])
        avg_max_pool = torch.cat([avg_pool, max_pool] + outputs, 1)
        out = avg_max_pool.view(avg_max_pool.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = nn.functional.relu(self.fc1(out))
        out = self.fc2(out)

        return out


class ClassificationModelDPNGap(nn.Module):
    def __init__(self, base_model, nb_features, num_init_features, dropout=0.5, gap_levels=(3, 7, 16, 27)):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.gap_levels = gap_levels

        self.fc1 = nn.Linear(nb_features, 256)
        self.fc2 = nn.Linear(256, config.NB_CATEGORIES)
        self.l1_4 = nn.Conv2d(4, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        # self.base_model.features['conv1_1']['conv'] = self.l1_4
        self.base_model.features[0].conv = self.l1_4

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        outputs = []

        def gap(x):
            if isinstance(x, tuple):
                return sum([gap(xi) for xi in x], [])
            else:
                x = x[:, :, 2:-2, 2:-2]  # discard borders
                return [F.avg_pool2d(x, x.shape[2:])]

        x = inputs
        for i, l in enumerate(self.base_model.features):
            x = l(x)
            # if isinstance(x, tuple):
            #     print(i, [xx.shape for xx in x])
            # else:
            #     print(i, x.shape)

            if i in self.gap_levels:
                outputs += gap(x)

        features = x

        avg_pool = F.avg_pool2d(features, features.shape[2:])
        max_pool = F.max_pool2d(features, features.shape[2:])
        avg_max_pool = torch.cat([avg_pool, max_pool] + outputs, 1)
        out = avg_max_pool.view(avg_max_pool.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = nn.functional.relu(self.fc1(out))
        out = self.fc2(out)
        # out = self.output_act(out)

        return out


class ClassificationModelDPNGwap(nn.Module):
    def __init__(self, base_model, nb_features, num_init_features,
                 dropout=0.5,
                 gwap_levels=(3, 7, 16, 27, 23),
                 gwap_channels=(100, 200, 300, 400, 832),
                 scale=1
                 ):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.gap_levels = gwap_levels

        self.gwap = {}
        for i, level in enumerate(gwap_levels):
            gwap = GWAP(gwap_channels[i], scale=scale)
            self.gwap[level] = gwap
            self.add_module(f'gwap{level}', gwap)

        self.fc1 = nn.Linear(nb_features, config.NB_CATEGORIES)
        self.l1_4 = nn.Conv2d(4, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.features[0].conv = self.l1_4

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        outputs = []

        def gwap(x, level):
            if isinstance(x, tuple):
                x = torch.cat(x, dim=1)
            res = self.gwap[level](x)
            return res

        x = inputs
        for i, l in enumerate(self.base_model.features):
            x = l(x)
            # if isinstance(x, tuple):
            #     print(i, [xx.shape for xx in x])
            # else:
            #     print(i, x.shape)

            if i in self.gap_levels:
                outputs += [gwap(x, i)]

        out = torch.cat(outputs, 1)
        out = out.view(out.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = self.fc1(out)
        # out = self.output_act(out)

        return out


def classification_model_resnet34(**kwargs):
    base_model = pretrainedmodels.resnet34()
    return ClassificationModelResnet(base_model, nb_features=512, **kwargs)


def classification_model_bninception(**kwargs):
    base_model = pretrainedmodels.bninception()
    return ClassificationModelBnInception(base_model, nb_features=1024, **kwargs)


def classification_model_bninception_wgap(**kwargs):
    base_model = pretrainedmodels.bninception()
    return ClassificationModelBnInceptionWGAP(base_model, **kwargs)


def classification_model_resnet18(**kwargs):
    base_model = pretrainedmodels.resnet18()
    return ClassificationModelResnet(base_model, nb_features=512, **kwargs)


def classification_model_resnet18_wgap(**kwargs):
    base_model = pretrainedmodels.resnet18()
    return ClassificationModelResnetWGAP(base_model, **kwargs)


def classification_model_resnet34_wgap(**kwargs):
    base_model = pretrainedmodels.resnet34()
    return ClassificationModelResnetWGAP(base_model, **kwargs)


def classification_model_resnet50(**kwargs):
    base_model = pretrainedmodels.resnet50()
    return ClassificationModelResnet(base_model, nb_features=2048, **kwargs)


def classification_model_resnet101(**kwargs):
    base_model = pretrainedmodels.resnet101()
    return ClassificationModelResnet(base_model, nb_features=2048, **kwargs)


def classification_model_se_resnext50(**kwargs):
    base_model = pretrainedmodels.se_resnext50_32x4d()
    return ClassificationModel(base_model, nb_features=2048, **kwargs)


def classification_model_se_resnext50_gap(**kwargs):
    base_model = pretrainedmodels.se_resnext50_32x4d()
    return ClassificationModelResNextGap(base_model, nb_features=5888, **kwargs)


def classification_model_se_resnext50_attn(**kwargs):
    base_model = pretrainedmodels.se_resnext50_32x4d()
    return ClassificationModelResNextAttn(base_model, nb_features=(2048, 1024, 512), **kwargs)


def classification_model_se_resnext101(**kwargs):
    base_model = pretrainedmodels.se_resnext101_32x4d()
    return ClassificationModel(base_model, nb_features=2048, **kwargs)


def classification_model_inceptionresnetv2(**kwargs):
    base_model = pretrainedmodels.inceptionresnetv2()
    return ClassificationModelInceptionResnetV2(base_model, nb_features=1536, **kwargs)


def classification_model_senet154(**kwargs):
    base_model = pretrainedmodels.senet154()
    return ClassificationModel(base_model, nb_features=2048, **kwargs)


def classification_model_dpn92(**kwargs):
    base_model = pretrainedmodels.dpn92()
    return ClassificationModelDPN(base_model, nb_features=2688, num_init_features=64, **kwargs)


def classification_model_dpn92_gap(**kwargs):
    base_model = pretrainedmodels.dpn92()
    return ClassificationModelDPNGap(base_model, num_init_features=64, **kwargs)


def classification_model_dpn68_gap(**kwargs):
    base_model = pretrainedmodels.dpn68b()
    return ClassificationModelDPNGap(base_model, num_init_features=10, **kwargs)


def classification_model_dpn68_gwap(**kwargs):
    base_model = pretrainedmodels.dpn68b()
    return ClassificationModelDPNGwap(base_model, num_init_features=10, **kwargs)


def classification_model_dpn_small_gwap(**kwargs):
    base_model = pretrainedmodels.models.dpn.DPN(
        small=True, num_init_features=10, k_r=128, groups=16,
        b=True, k_sec=(3, 4, 4, 3), inc_sec=(16, 32, 32, 64),
        num_classes=config.NB_CATEGORIES, test_time_pool=True)

    return ClassificationModelDPNGwap(base_model, num_init_features=10, **kwargs)


def classification_model_dpn92_gwap(**kwargs):
    base_model = pretrainedmodels.dpn92()
    return ClassificationModelDPNGwap(base_model, num_init_features=64, **kwargs)


def classification_model_dpn107(**kwargs):
    base_model = pretrainedmodels.dpn107()
    return ClassificationModelDPN(base_model, nb_features=2688, num_init_features=128, **kwargs)


def classification_model_xception(**kwargs):
    base_model = pretrainedmodels.xception(pretrained=False)

    state_dict = torch.utils.model_zoo.load_url(
        pretrainedmodels.pretrained_settings['xception']['imagenet']['url'],
        model_dir='models')
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    base_model.load_state_dict(state_dict, strict=False)

    return ClassificationModel(base_model, nb_features=2048, **kwargs)


class ClassificationModelSeResnextWGAP(nn.Module):
    def __init__(self,
                 base_model,
                 inplanes,
                 dropout=0.5,
                 gwap_channels=(64, 128, 256, 512),
                 scale=1,
                 fc_layers=256,
                 trim_layers=0
                 ):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.trim_layers = trim_layers

        self.gwap0 = GWAP(gwap_channels[0], scale=scale)
        self.gwap1 = GWAP(gwap_channels[1], scale=scale)
        self.gwap2 = GWAP(gwap_channels[2], scale=scale)
        self.gwap3 = GWAP(gwap_channels[3], scale=scale)

        self.fc_out_0 = nn.Linear(sum(gwap_channels), fc_layers)
        self.fc_out_1 = nn.Linear(fc_layers, config.NB_CATEGORIES)

        self.l1_4 = nn.Conv2d(4, inplanes, kernel_size=7, stride=2,
                              padding=3, bias=False)

        self.base_model.layer0[0] = self.l1_4

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        outputs = []

        x = self.base_model.layer0(inputs)
        x = self.base_model.layer1(x)
        outputs.append(self.gwap0(x))
        x = self.base_model.layer2(x)
        outputs.append(self.gwap1(x))

        if self.trim_layers < 2:
            x = self.base_model.layer3(x)
            outputs.append(self.gwap2(x))

        if self.trim_layers < 1:
            x = self.base_model.layer4(x)
            outputs.append(self.gwap3(x))

        out = torch.cat(outputs, 1)
        out = out.view(out.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = self.fc_out_0(out)
        out = F.relu(out)
        out = self.fc_out_1(out)
        return out


def classification_model_se_resnext50_gwap(**kwargs):
    base_model = pretrainedmodels.se_resnext50_32x4d()
    return ClassificationModelSeResnextWGAP(base_model, inplanes=64, **kwargs)


class ClassificationModelResnet34WGAP(nn.Module):
    def __init__(self,
                 dropout=0.5,
                 gwap_channels=(64, 128, 256, 512),
                 scale=1,
                 fc_layers=(888, 256),
                 extra_gwap_layer=None
                 ):
        super().__init__()
        self.dropout = dropout
        inplanes=64

        self.base_model = pretrainedmodels.models.senet.SENet(
            pretrainedmodels.models.senet.SEResNeXtBottleneck, [3, 3, 3, 2], groups=8, reduction=16,
            dropout_p=None, inplanes=inplanes, input_3x3=False,
            downsample_kernel_size=1, downsample_padding=0,
            num_classes=config.NB_CATEGORIES
        )

        layer0_modules = [
            ('conv1', nn.Conv2d(4, 64, 3, stride=2, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(inplanes)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True))
        ]

        self.base_model.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        self.gwap = []
        for i, ch in enumerate(gwap_channels):
            gwap = GWAP(ch, scale=scale, extra_layer=extra_gwap_layer)
            self.gwap.append(gwap)
            self.add_module(f'gwap{i}', gwap)

        self.fc_layers = []
        fc_layers = list(fc_layers) + [config.NB_CATEGORIES]
        for i in range(len(fc_layers)-1):
            fc = nn.Linear(fc_layers[i], fc_layers[i+1])
            self.fc_layers.append(fc)
            self.add_module(f'fc_out_{i}', fc)

    def freeze_encoder(self):
        pass

    def unfreeze_encoder(self):
        pass

    def forward(self, inputs):
        outputs = []

        x = self.base_model.layer0(inputs)
        x = self.base_model.layer1(x)
        outputs.append(self.gwap[0](x))
        x = self.base_model.layer2(x)
        outputs.append(self.gwap[1](x))
        x = self.base_model.layer3(x)
        outputs.append(self.gwap[2](x))
        x = self.base_model.layer4(x)
        outputs.append(self.gwap[3](x))

        out = torch.cat(outputs, 1)
        out = out.view(out.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out_act = out
        for fc in self.fc_layers:
            out = fc(out_act)
            out_act = F.relu(out)
        return out


def classification_model_resnext34_wgap(**kwargs):
    return ClassificationModelResnet34WGAP(**kwargs)


import pytorchcv.models.airnet
import pytorchcv.models.airnext
import pytorchcv.models.common


class ClassificationModelAirNeXtWGAP(nn.Module):
    def __init__(self, base_model,
                 dropout=0.5,
                 gwap_levels=(0, 1, 2, 3, 4),
                 gwap_channels=(64, 256, 512, 1024, 2048),
                 fc_layers=(256,),
                 scale=3):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.gap_levels = gwap_levels

        self.gwap = {}
        for i, level in enumerate(gwap_levels):
            gwap = GWAP(gwap_channels[i], scale=scale)
            self.gwap[level] = gwap
            self.add_module(f'gwap{level}', gwap)

        self.l1_4 = pytorchcv.models.common.conv3x3_block(
            in_channels=4,
            out_channels=32,
            stride=2)
        self.base_model.features[0].conv1 = self.l1_4

        self.fc_layers = []
        fc_layers = [sum(gwap_channels)] + list(fc_layers) + [config.NB_CATEGORIES]
        for i in range(len(fc_layers) - 1):
            fc = nn.Linear(fc_layers[i], fc_layers[i + 1])
            self.fc_layers.append(fc)
            self.add_module(f'fc_out_{i}', fc)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        outputs = []

        x = inputs
        for i, l in enumerate(self.base_model.features):
            x = l(x)
            # print(i, x.shape)

            if i in self.gap_levels:
                outputs += [self.gwap[i](x)]

        out = torch.cat(outputs, 1)
        out = out.view(out.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out_act = out
        for fc in self.fc_layers:
            out = fc(out_act)
            out_act = F.relu(out)
        return out


def classification_model_airnext50_wgap(**kwargs):
    base_model = pytorchcv.models.airnext.airnext50_32x4d_r2(pretrained=True)
    return ClassificationModelAirNeXtWGAP(base_model=base_model, **kwargs)
