import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import pretrainedmodels
from pytorchcv.model_provider import get_model as ptcv_get_model


class Xception_osmr(nn.Module):

    def __init__(self, classCount):
        super(Xception_osmr, self).__init__()

        self.model_ft = ptcv_get_model("xception", pretrained=True)
        num_ftrs = self.model_ft.output.in_features
        self.model_ft.features.final_block.pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.output = nn.Sequential(nn.Linear(num_ftrs, classCount, bias=True),
                                             nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x

class se_resnext50_32x4d(nn.Module):

    def __init__(self, classCount):
        super(se_resnext50_32x4d, self).__init__()

        self.model_ft = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, classCount, bias=True),
                                             nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x

class DenseNet169_change_avg(nn.Module):

    def __init__(self, classCount, isTrained=False):
    
        super(DenseNet169_change_avg, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1664, classCount)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):

        x = self.densenet169(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        
        return x


class DenseNet121_change_avg(nn.Module):

    def __init__(self, classCount, isTrained=False):
    
        super(DenseNet121_change_avg, self).__init__()
        
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, classCount)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):

        x = self.densenet121(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)
        x = self.sigmoid(x)
        
        return x

class ibn_densenet121_osmr(nn.Module):

    def __init__(self, classCount):
        super(ibn_densenet121_osmr, self).__init__()

        self.model_ft = ptcv_get_model("ibn_densenet121", pretrained=True)
        num_ftrs = self.model_ft.output.in_features
        self.model_ft.features.final_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.output = nn.Sequential(nn.Linear(num_ftrs, classCount, bias=True),
                                             nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x  


def get_model(backbone):

    if backbone == 'DenseNet121_change_avg':
        model = DenseNet121_change_avg(28, True)
    elif backbone == 'ibn_densenet121_osmr':
        model = ibn_densenet121_osmr(28)
    elif backbone == 'DenseNet169_change_avg':
        model = DenseNet169_change_avg(28, True)
    elif backbone == 'se_resnext50_32x4d':
        model = se_resnext50_32x4d(28)
    elif backbone == 'Xception_osmr':
        model = Xception_osmr(28)

    return model