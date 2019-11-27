from config.config import *
from utils.common_util import *
from networks.resnet_ml import class_resnet50_dropout

model_names = {
    'class_resnet50_dropout': 'resnet50-19c8e357.pth',
}

def init_network(params):
    architecture = params.get('architecture', 'class_resnet50_dropout')
    num_classes = params.get('num_classes', 12815)
    in_channels = params.get('in_channels', 4)

    pretrained_file = opj(PRETRAINED_DIR, model_names[architecture])
    print(">> Using pre-trained model.")
    net = eval(architecture)(num_classes=num_classes, in_channels=in_channels, pretrained_file=pretrained_file)
    return net
