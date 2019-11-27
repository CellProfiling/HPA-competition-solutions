import torch
from torch import nn
from torch.nn import functional as F
from .pytorch_mobilenet_v2_xzhu import MobileNetV2
from .xception import Xception
import pretrainedmodels

from pretrainedmodels import xception


class MyXception(nn.Module):

    def __init__(self, config, pretrained='imagenet', device='cuda', no_fc=False):
        super().__init__()
        self.device = device
        self.base_model = Xception(n_input_channels=config['n_channels'])

        if no_fc == 0:
            self.fc = nn.Sequential()  # identity layer
            self.postprocessing = nn.Sequential()
        else:
            self.fc = nn.Linear(2048, config['_n_classes'])
            self.postprocessing = nn.Sigmoid()

        if pretrained == 'imagenet':
            w = torch.load('pretrained_models/xception_features.pth')
            self.base_model.features.load_state_dict(w)
            w = torch.load('pretrained_models/xception_conv1.pth')
            if self.base_model.n_input_channels == 4:
                w['weight'] = w['weight'][:, [0, 1, 2, 0], :, :]
            elif self.base_model.n_input_channels == 1:
                w['weight'] = w['weight'][:, [0], :, :]
            self.base_model.conv1.load_state_dict(w)

    def forward(self, x):
        x = self.base_model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.postprocessing(x)
        return x


class MyMobileNetV2(nn.Module):

    def __init__(self, config, device='cuda', pretrained='imagenet'):
        super().__init__()
        self.device = device
        self.base_model = MobileNetV2(
            n_input_channels=config['n_channels'],
            n_class=config['_n_classes'],
            input_size=config['size'],
        )

        self.postprocessing = nn.Sigmoid()

        if pretrained == 'imagenet':
            entrance_weights = torch.load('pretrained_models/mobilenet_v2_entrance.pt', self.device)
            entrance_weights['0.weight'] = entrance_weights['0.weight'][:, [0, 1, 2, 0], :, :]
            self.base_model.entrance.load_state_dict(entrance_weights)
            self.base_model.ir_blocks.load_state_dict(
                torch.load('pretrained_models/mobilenet_v2_ir_blocks.pt', self.device)
            )

    def forward(self, x):
        x = self.base_model(x)
        x = self.postprocessing(x)
        return x
