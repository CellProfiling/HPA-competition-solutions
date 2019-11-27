from .se_resnet import se_resnext50_32x4d
from torch import nn
import torch
from torch.nn import functional as F

class MaxAvgPool(nn.Module):

    def __init__(self):
        super(MaxAvgPool, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)

    def forward(self, x):
        batch_size = x.shape[0]
        f = self.avgpool(x).view(batch_size, -1)+self.maxpool(x).view(batch_size, -1)
        return f

class Model(nn.Module):
    '''
    concat in decoder
    '''
    def __init__(self):
        super(Model, self).__init__()

        self.resnet = se_resnext50_32x4d()
        self.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        w = self.resnet.layer0.conv1.weight
        self.conv0.weight = nn.Parameter(torch.cat((w, (w[:, :1, :, :] + w[:, 2:, :, :]) * 0.5), dim=1))

        self.down0 = nn.Sequential(
            self.conv0,
            self.resnet.layer0.bn1,
            self.resnet.layer0.relu1,
            self.resnet.layer0.pool
        )

        self.down1 = self.resnet.layer1  # 64, 256
        self.down2 = self.resnet.layer2  # 32, 512
        self.down3 = self.resnet.layer3  # 16, 1024
        self.down4 = self.resnet.layer4  # 8, 2048

        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)

        self.logit = nn.Sequential(
            MaxAvgPool(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 28)
        )

        self.logit_64 = nn.Sequential(
            MaxAvgPool(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 28)
        )

        self.logit_128 = nn.Sequential(
            MaxAvgPool(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 28)
        )

        self.logit_256 = nn.Sequential(
            MaxAvgPool(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 28)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = torch.cat([
            (x[:, [0]] - mean[0]) / std[0],
            (x[:, [1]] - mean[1]) / std[1],
            (x[:, [2]] - mean[2]) / std[2],
            x[:, [3]],
        ], 1)

        x = self.down0(x)
        x = self.down1(x)

        logit_64  = self.logit_64(x)

        x = self.down2(x)

        logit_128 = self.logit_128(x)
        x = self.down3(x)

        logit_256 = self.logit_256(x)

        x = self.down4(x)

        logit = self.logit(x)

        return logit, logit_64, logit_128, logit_256


if __name__ == '__main__':
    net = Model()
    x = torch.rand(4, 4, 224, 224)
    net.forward(x)

