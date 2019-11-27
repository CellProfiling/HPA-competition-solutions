import re

from layers.backbone.densenet import *
from layers.loss import *


## networks  ######################################################################
class DensenetClass(nn.Module):
    def load_pretrain(self, pretrain_file):

        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrain_file)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.backbone.load_state_dict(state_dict)

    def __init__(self,feature_net='densenet121', num_classes=28,
                 in_channels=3,
                 pretrained_file=None,
                 dropout=False,
                 large=False,
                 ):
        super().__init__()
        self.dropout = dropout
        self.in_channels = in_channels
        self.large = large

        if feature_net=='densenet121':
            self.backbone = densenet121()
            num_features = 1024
        elif feature_net=='densenet169':
            self.backbone = densenet169()
            num_features = 1664
        elif feature_net=='densenet161':
            self.backbone = densenet161()
            num_features = 2208
        elif feature_net=='densenet201':
            self.backbone = densenet201()
            num_features = 1920

        self.load_pretrain(pretrained_file)

        if self.in_channels > 3:
            # https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
            w = self.backbone.features.conv0.weight
            self.backbone.features.conv0 = nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3, 3), bias=False)
            self.backbone.features.conv0.weight = torch.nn.Parameter(torch.cat((w, w[:,:1,:,:]),dim=1))

        self.conv1 =nn.Sequential(
            self.backbone.features.conv0,
            self.backbone.features.norm0,
            self.backbone.features.relu0,
            self.backbone.features.pool0
        )
        self.encoder2 = nn.Sequential(self.backbone.features.denseblock1,
                                      )
        self.encoder3 = nn.Sequential(self.backbone.features.transition1,
                                      self.backbone.features.denseblock2,
                                      )
        self.encoder4 = nn.Sequential(self.backbone.features.transition2,
                                      self.backbone.features.denseblock3,
                                      )
        self.encoder5 = nn.Sequential(self.backbone.features.transition3,
                                      self.backbone.features.denseblock4,
                                      self.backbone.features.norm5)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.logit = nn.Linear(num_features, num_classes)

        # https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        if self.dropout:
            self.bn1 = nn.BatchNorm1d(num_features*2)
            self.fc1 = nn.Linear(num_features*2, num_features)
            self.bn2 = nn.BatchNorm1d(num_features)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        mean = [0.074598, 0.050630, 0.050891, 0.076287]#rgby
        std =  [0.122813, 0.085745, 0.129882, 0.119411]
        for i in range(self.in_channels):
            x[:,i,:,:] = (x[:,i,:,:] - mean[i]) / std[i]

        x = self.conv1(x)
        if self.large:
            x = self.maxpool(x)
        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        # print(e2.shape, e3.shape, e4.shape, e5.shape)
        e5 = F.relu(e5,inplace=True)
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
        x = x.view(x.size(0), -1)
        x = self.logit(x)
        return x

def class_densenet121_dropout(**kwargs):
    num_classes = kwargs['num_classes']
    in_channels = kwargs['in_channels']
    pretrained_file = kwargs['pretrained_file']
    model = DensenetClass(feature_net='densenet121', num_classes=num_classes,
                        in_channels=in_channels, pretrained_file=pretrained_file, dropout=True)
    return model

def class_densenet121_large_dropout(**kwargs):
    num_classes = kwargs['num_classes']
    in_channels = kwargs['in_channels']
    pretrained_file = kwargs['pretrained_file']
    model = DensenetClass(feature_net='densenet121', num_classes=num_classes,
                        in_channels=in_channels, pretrained_file=pretrained_file, dropout=True, large=True)
    return model
