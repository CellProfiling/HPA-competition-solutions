from torch.nn import Parameter

from layers.backbone.resnet import *
from layers.loss import *
from utils.common_util import *


## networks  ######################################################################

# https://github.com/ronghuaiyang/arcface-pytorch
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        return cosine


class ResnetClass(nn.Module):

    def __init__(self,
                 feature_net='resnet34',
                 num_classes=28,
                 in_channels=3,
                 attention_type='',
                 pretrained_file=None,
                 dropout=False,
                 ):
        super().__init__()
        self.attention_type=attention_type
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

        self.resnet.load_state_dict(torch.load(pretrained_file, map_location=lambda storage, loc: storage))
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
            self.resnet.maxpool
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(512 * self.EX, num_classes)
        self.arc_margin_product=ArcMarginProduct(512, num_classes)
        # https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        if self.dropout:
            self.bn1 = nn.BatchNorm1d(1024 * self.EX)
            self.fc1 = nn.Linear(1024 * self.EX, 512 * self.EX)
            self.bn2 = nn.BatchNorm1d(512 * self.EX)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(512 * self.EX, 512)
            self.bn3 = nn.BatchNorm1d(512)
        self.extract_feature = False

    def set_configs(self, extract_feature=False, **kwargs):
        self.extract_feature = extract_feature

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        # print(x.shape, e2.shape,e3.shape,e4.shape,e5.shape)
        if self.dropout:
            x = torch.cat((nn.AdaptiveAvgPool2d(1)(e5), nn.AdaptiveMaxPool2d(1)(e5)), dim=1)
            x = x.view(x.size(0), -1)
            x = self.bn1(x)
            x = F.dropout(x, p=0.25)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.bn2(x)
            x = F.dropout(x, p=0.5)
        else:
            x = self.avgpool(e5)
        x = x.view(x.size(0), -1)

        x = self.fc2(x)
        feature = self.bn3(x)

        # logits = self.logit(feature)
        cosine=self.arc_margin_product(feature)
        if self.extract_feature:
            return cosine, feature
        else:
            return cosine

def class_resnet50_dropout(**kwargs):
    num_classes = kwargs['num_classes']
    in_channels = kwargs['in_channels']
    pretrained_file = kwargs['pretrained_file']
    model = ResnetClass(feature_net='resnet50', num_classes=num_classes,
                        in_channels=in_channels, pretrained_file=pretrained_file, dropout=True)
    return model
