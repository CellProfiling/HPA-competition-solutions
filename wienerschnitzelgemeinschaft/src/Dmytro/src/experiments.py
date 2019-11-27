import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels
import config

from models import *

class ModelInfo:
    def __init__(self,
                 factory,
                 pretrained_settings,
                 args,
                 batch_size,
                 dataset_args,
                 optimiser='adam',
                 initial_lr=1e-5,
                 loss='bce',
                 scheduler='cos',
                 crop_size=512,
                 nb_epochs=45,
                 is_pretrained=True,
                 folds_split='orig'
                 ):
        self.nb_epochs = nb_epochs
        self.initial_lr = initial_lr
        self.factory = factory
        self.pretrained_settings = pretrained_settings
        self.args = args
        self.batch_size = batch_size
        self.dataset_args = dataset_args
        self.optimiser = optimiser
        self.loss = loss
        self.scheduler = scheduler
        self.crop_size = crop_size
        self.is_pretrained = is_pretrained
        self.folds_split = folds_split


MODELS = {
    'se_resnext50': ModelInfo(
        classification_model_se_resnext50,
        pretrained_settings=pretrainedmodels.pretrained_settings['se_resnext50_32x4d']['imagenet'],
        args={},
        batch_size=8,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=0)
    ),
    'se_resnext50_ia10': ModelInfo(
        classification_model_se_resnext50,
        pretrained_settings=pretrainedmodels.pretrained_settings['se_resnext50_32x4d']['imagenet'],
        args={},
        batch_size=8,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=10)
    ),
    'se_resnext50_ia20': ModelInfo(
        classification_model_se_resnext50,
        pretrained_settings=pretrainedmodels.pretrained_settings['se_resnext50_32x4d']['imagenet'],
        args={},
        batch_size=8,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20)
    ),
    'se_resnext50_gap_ia20': ModelInfo(
        classification_model_se_resnext50_gap,
        pretrained_settings=pretrainedmodels.pretrained_settings['se_resnext50_32x4d']['imagenet'],
        args={},
        batch_size=8,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20)
    ),
    'se_resnext50_gap_ia20_sgd_fl': ModelInfo(
        classification_model_se_resnext50_gap,
        pretrained_settings=pretrainedmodels.pretrained_settings['se_resnext50_32x4d']['imagenet'],
        args=dict(),
        batch_size=8,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.05,
        loss='focal_loss'
    ),
    'se_resnext50_attn234': ModelInfo(
        classification_model_se_resnext50_attn,
        pretrained_settings=pretrainedmodels.pretrained_settings['se_resnext50_32x4d']['imagenet'],
        args=dict(attn_levels=(2, 3, 4)),
        batch_size=8,
        dataset_args={}
    ),
    'se_resnext50_attn34': ModelInfo(
        classification_model_se_resnext50_attn,
        pretrained_settings=pretrainedmodels.pretrained_settings['se_resnext50_32x4d']['imagenet'],
        args=dict(attn_levels=(3, 4)),
        batch_size=8,
        dataset_args={}
    ),
    'se_resnext50_attn4': ModelInfo(
        classification_model_se_resnext50_attn,
        pretrained_settings=pretrainedmodels.pretrained_settings['se_resnext50_32x4d']['imagenet'],
        args=dict(attn_levels=(4,)),
        batch_size=8,
        dataset_args={}
    ),
    'resnet34': ModelInfo(
        classification_model_resnet34,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet34']['imagenet'],
        args={},
        batch_size=32,
        dataset_args={}
    ),
    'resnet18': ModelInfo(
        classification_model_resnet18,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet18']['imagenet'],
        args=dict(dropout=0),
        batch_size=64,
        nb_epochs=256,
        optimiser='sgd',
        initial_lr=0.1,
        loss='focal_loss',
        dataset_args={}
    ),
    'resnet18_gwap': ModelInfo(
        classification_model_resnet18_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet18']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(960, 256),
                  scale=3,
                  dropout=0.25),
        batch_size=64,
        nb_epochs=256,
        optimiser='sgd',
        initial_lr=0.1,
        loss='focal_loss',
        dataset_args={}
    ),
    'resnet18_gwap_el_64': ModelInfo(
        classification_model_resnet18_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet18']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(960, 256),
                  scale=3,
                  dropout=0.25,
                  extra_gwap_layer=64
                  ),
        batch_size=48,
        nb_epochs=256,
        optimiser='sgd',
        initial_lr=0.1,
        loss='focal_loss',
        dataset_args={}
    ),
    'resnet18_gwap_no_dropout': ModelInfo(
        classification_model_resnet18_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet18']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(960, 256),
                  scale=3,
                  dropout=0),
        batch_size=64,
        nb_epochs=256,
        optimiser='sgd',
        initial_lr=0.1,
        loss='focal_loss',
        dataset_args={}
    ),
    'resnet34_wgap3': ModelInfo(
        classification_model_resnet34_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet34']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(960, 256),
                  scale=3),
        batch_size=32,
        nb_epochs=64,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.1,
        loss='focal_loss'
    ),
    'resnet34_wgap3_bce_f1': ModelInfo(
        classification_model_resnet34_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet34']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(960, 256),
                  scale=3),
        batch_size=32,
        nb_epochs=64,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.1,
        loss='bce_f1'
    ),
    'resnet34_wgap3_bce': ModelInfo(
        classification_model_resnet34_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet34']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(960, 256),
                  scale=3),
        batch_size=32,
        nb_epochs=64,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.1,
        loss='bce'
    ),
    'se_resnet34_wgap3': ModelInfo(
        classification_model_resnext34_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet34']['imagenet'],
        args=dict(gwap_channels=(256, 512, 1024, 2048),
                  fc_layers=(sum((256, 512, 1024, 2048)), 256),
                  scale=3),
        batch_size=16,
        nb_epochs=64,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.1,
        loss='focal_loss'
    ),
    'resnet34_wgap3_no_dropout': ModelInfo(
        classification_model_resnet34_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet34']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(960, 256),
                  scale=3,
                  dropout=0
                  ),
        batch_size=32,
        nb_epochs=160,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.1,
        loss='focal_loss'
    ),
    'resnet34_wgap3_step': ModelInfo(
        classification_model_resnet34_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet34']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(960, 256),
                  scale=3),
        batch_size=32,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.01,
        loss='focal_loss',
        scheduler='step'
    ),
    'resnet34_wgap3_step_1024': ModelInfo(
        classification_model_resnet34_wgap,
        crop_size=1024,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet34']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(960, 256),
                  scale=3),
        batch_size=8,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.01,
        loss='focal_loss',
        scheduler='step'
    ),
    'resnet50': ModelInfo(
        classification_model_resnet50,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet34']['imagenet'],
        args={},
        batch_size=20,
        dataset_args={}
    ),
    'resnet101': ModelInfo(
        classification_model_resnet101,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet101']['imagenet'],
        args={},
        batch_size=12,
        dataset_args={}
    ),
    'se_resnext101': ModelInfo(
        classification_model_se_resnext101,
        pretrained_settings=pretrainedmodels.pretrained_settings['se_resnext101_32x4d']['imagenet'],
        args={},
        batch_size=6,
        dataset_args={}
    ),
    'inceptionresnetv2': ModelInfo(
        classification_model_inceptionresnetv2,
        pretrained_settings=pretrainedmodels.pretrained_settings['inceptionresnetv2']['imagenet'],
        args={},
        batch_size=8,
        dataset_args={}
    ),
    'dpn92': ModelInfo(
        classification_model_dpn92,
        pretrained_settings=pretrainedmodels.pretrained_settings['inceptionresnetv2']['imagenet'],
        args={},
        batch_size=8,
        dataset_args={}
    ),
    'dpn92_ia10': ModelInfo(
        classification_model_dpn92,
        pretrained_settings=pretrainedmodels.pretrained_settings['inceptionresnetv2']['imagenet'],
        args={},
        batch_size=8,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=10)
    ),
    'dpn92_ia20': ModelInfo(
        classification_model_dpn92,
        pretrained_settings=pretrainedmodels.pretrained_settings['inceptionresnetv2']['imagenet'],
        args={},
        batch_size=8,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20)
    ),
    'dpn92_gap_3.7.16.27_ia20': ModelInfo(
        classification_model_dpn92_gap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(gap_levels=(3, 7, 16, 27), nb_features=9256),
        batch_size=8,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20)
    ),
    'dpn92_wgap': ModelInfo(
        classification_model_dpn92_gwap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(gwap_levels=(3, 7, 16, 27, 31),
                  gwap_channels=(336, 704, 1288, 1552, 2688),
                  nb_features=6568,
                  scale=1),
        batch_size=8,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.05,
        loss='focal_loss'
    ),
    'dpn68_gap_ia20': ModelInfo(
        classification_model_dpn68_gap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(gap_levels=(3, 7, 12, 19), nb_features=3312),
        batch_size=16,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20)
    ),
    'dpn68_gap_ia20_sgd': ModelInfo(
        classification_model_dpn68_gap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(gap_levels=(3, 7, 12, 19), nb_features=3312),
        batch_size=16,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.03
    ),
    'dpn68_gap_ia20_sgd_fl': ModelInfo(
        classification_model_dpn68_gap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(gap_levels=(3, 7, 12, 19), nb_features=3312),
        batch_size=16,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.05,
        loss='focal_loss'
    ),
    'dpn68_gwap_ia20_sgd_fl': ModelInfo(
        classification_model_dpn68_gwap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(gwap_levels=(3, 7, 12, 19, 23), gwap_channels=(144, 320, 480, 704, 832), nb_features=2480),
        batch_size=16,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.05,
        loss='focal_loss'
    ),
    'dpn68_gwap3_ia20_sgd_fl': ModelInfo(
        classification_model_dpn68_gwap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(gwap_levels=(3, 7, 12, 19, 23), gwap_channels=(144, 320, 480, 704, 832), nb_features=2480, scale=3),
        batch_size=16,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.05,
        loss='focal_loss',
    ),
    'dpn_small_gwap3': ModelInfo(
        classification_model_dpn_small_gwap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(gwap_levels=(3, 7, 11, 15),
                  gwap_channels=(64+80, 128+192,
                                 256+192, 512+320),
                  nb_features=1744, scale=3),
        batch_size=16,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.05,
        loss='focal_loss',
        is_pretrained=False
    ),
    'dpn68_gwap_ia20_sgd_fl2': ModelInfo(
        classification_model_dpn68_gwap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(gwap_levels=(3, 7, 12, 19, 23), gwap_channels=(144, 320, 480, 704, 832), nb_features=2480, scale=1),
        batch_size=16,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.05,
        loss='focal_loss2'
    ),
    'dpn68_gwap_ia20_sgd_fl_1024': ModelInfo(
        classification_model_dpn68_gwap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(gwap_levels=(3, 7, 12, 19, 23), gwap_channels=(144, 320, 480, 704, 832), nb_features=2480),
        batch_size=4,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.05,
        loss='focal_loss'
    ),
    'dpn107': ModelInfo(
        classification_model_dpn107,
        pretrained_settings=pretrainedmodels.pretrained_settings['inceptionresnetv2']['imagenet'],
        args={},
        batch_size=4,
        dataset_args={}
    ),
    'dpn107_ia20': ModelInfo(
        classification_model_dpn107,
        pretrained_settings=pretrainedmodels.pretrained_settings['inceptionresnetv2']['imagenet'],
        args={},
        batch_size=4,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20)
    ),
    'senet154': ModelInfo(
        classification_model_senet154,
        pretrained_settings=pretrainedmodels.pretrained_settings['senet154']['imagenet'],
        args={},
        batch_size=8,
        dataset_args={}
    ),
    'xception': ModelInfo(
        classification_model_xception,
        pretrained_settings=pretrainedmodels.pretrained_settings['xception']['imagenet'],
        args={},
        batch_size=16,
        dataset_args={}
    ),
    'bn_inception': ModelInfo(
        classification_model_bninception,
        pretrained_settings=pretrainedmodels.pretrained_settings['bninception']['imagenet'],
        args=dict(dropout=0),
        batch_size=32,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.1,
        loss='bce',
        nb_epochs=256
    ),
    'bn_inception_wgap': ModelInfo(
        classification_model_bninception_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['bninception']['imagenet'],
        args=dict(
            gwap_channels=(192, 576, 1056, 1024),
            fc_layers=(2848, 256),
            scale=3,
            dropout=0.0),
        batch_size=32,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.1,
        loss='bce',
        nb_epochs=256
    ),
    'bn_inception_wgap_1024': ModelInfo(
        classification_model_bninception_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['bninception']['imagenet'],
        args=dict(
            gwap_channels=(192, 576, 1056, 1024),
            fc_layers=(2848, 256),
            scale=3,
            dropout=0.0),
        batch_size=16,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20),
        optimiser='sgd',
        initial_lr=0.1,
        loss='bce',
        nb_epochs=256,
        crop_size=1024
    ),
    'resnet18_gwap_aug_20_30': ModelInfo(
        classification_model_resnet18_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet18']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(960, 256),
                  scale=3,
                  dropout=0),
        batch_size=64,
        nb_epochs=256,
        optimiser='sgd',
        initial_lr=0.1,
        loss='focal_loss',
        dataset_args=dict(geometry_aug_level=20, img_aug_level=30)
    ),
    'resnet34_gwap_aug_20_30': ModelInfo(
        classification_model_resnet34_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet18']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(960, 256),
                  scale=3,
                  dropout=0),
        batch_size=32,
        nb_epochs=256,
        optimiser='sgd',
        initial_lr=0.1,
        loss='focal_loss',
        dataset_args=dict(geometry_aug_level=20, img_aug_level=30)
    ),
    'resnet34_gwap_trim1': ModelInfo(
        classification_model_resnet34_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet18']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(448, 256),
                  scale=3,
                  dropout=0,
                  trim_layers=1
                  ),
        batch_size=32,
        nb_epochs=256,
        optimiser='sgd',
        initial_lr=0.1,
        loss='bce',
        dataset_args=dict(geometry_aug_level=20, img_aug_level=30)
    ),
    'resnet34_gwap_trim2': ModelInfo(
        classification_model_resnet34_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet18']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(192, 256),
                  scale=3,
                  dropout=0,
                  trim_layers=2
                  ),
        batch_size=32,
        nb_epochs=256,
        optimiser='sgd',
        initial_lr=0.1,
        loss='bce',
        dataset_args=dict(geometry_aug_level=20, img_aug_level=30)
    ),
    'dpn68_gwap_extra': ModelInfo(
        classification_model_dpn68_gwap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(
            gwap_levels=(3, 7, 12, 19, 23),
            gwap_channels=(144, 320, 480, 704, 832),
            nb_features=2480,
            scale=3,
            dropout=0.5),
        batch_size=16,
        dataset_args=dict(geometry_aug_level=20, img_aug_level=30, use_extarnal=True, folds_split='emb'),
        optimiser='sgd',
        initial_lr=0.05,
        loss='bce'
    ),
    'dpn68_gwap_extra_trim1': ModelInfo(
        classification_model_dpn68_gwap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(
            gwap_levels=(3, 7, 12, 19),
            gwap_channels=(144, 320, 480, 704),
            nb_features=1648,
            scale=3,
            dropout=0.5),
        batch_size=16,
        dataset_args=dict(geometry_aug_level=20, img_aug_level=30, use_extarnal=True),
        optimiser='sgd',
        initial_lr=0.05,
        loss='bce',
        nb_epochs=80
    ),
    'dpn68_gwap_extra_cluster4x': ModelInfo(
        classification_model_dpn68_gwap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(
            gwap_levels=(3, 7, 12, 19, 23),
            gwap_channels=(144, 320, 480, 704, 832),
            nb_features=2480,
            scale=3,
            dropout=0.5),
        batch_size=16,
        dataset_args=dict(geometry_aug_level=20, img_aug_level=30, use_extarnal=True),
        folds_split='cluster4x',
        optimiser='sgd',
        initial_lr=0.05,
        loss='bce'
    ),
    'dpn68_gwap_extra_emb': ModelInfo(
        classification_model_dpn68_gwap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(
            gwap_levels=(3, 7, 12, 19, 23),
            gwap_channels=(144, 320, 480, 704, 832),
            nb_features=2480,
            scale=3,
            dropout=0.5),
        batch_size=16,
        dataset_args=dict(geometry_aug_level=20, img_aug_level=30, use_extarnal=True),
        folds_split='emb',
        optimiser='sgd',
        initial_lr=0.05,
        loss='bce'
    ),
    'resnet34_gwap_extra_cluster4x': ModelInfo(
        classification_model_resnet34_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet18']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(960, 256),
                  scale=3,
                  dropout=0),
        batch_size=32,
        nb_epochs=256,
        optimiser='sgd',
        initial_lr=0.1,
        loss='focal_loss',
        dataset_args=dict(geometry_aug_level=20, img_aug_level=30, use_extarnal=True),
        folds_split='cluster4x',
    ),
    'resnet34_gwap_extra_emb': ModelInfo(
        classification_model_resnet34_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet18']['imagenet'],
        args=dict(gwap_channels=(64, 128, 256, 512),
                  fc_layers=(960, 256),
                  scale=3,
                  dropout=0),
        batch_size=32,
        nb_epochs=256,
        optimiser='sgd',
        initial_lr=0.1,
        loss='focal_loss',
        dataset_args=dict(geometry_aug_level=20, img_aug_level=30, use_extarnal=True),
        folds_split='emb',
    ),
    'airnext50_gwap_extra_emb': ModelInfo(
        classification_model_airnext50_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['dpn92']['imagenet+5k'],
        args=dict(
            gwap_levels=(0, 1, 2, 3, 4),
            gwap_channels=(64, 256, 512, 1024, 2048),
            fc_layers=(256,),
            scale=3,
            dropout=0.5),
        batch_size=8,
        dataset_args=dict(geometry_aug_level=20, img_aug_level=30, use_extarnal=True),
        folds_split='emb',
        optimiser='sgd',
        initial_lr=0.05,
        loss='bce'
    ),
    'se_resnext50_gwap_extra_emb': ModelInfo(
        classification_model_se_resnext50_gwap,
        pretrained_settings=pretrainedmodels.pretrained_settings['resnet18']['imagenet'],
        args=dict(
            gwap_channels=(256, 512, 1024, 2048),
            fc_layers=256,
            scale=3,
            dropout=0.5),
        batch_size=8,
        dataset_args=dict(geometry_aug_level=20, img_aug_level=30, use_extarnal=True),
        folds_split='emb',
        optimiser='sgd',
        initial_lr=0.05,
        loss='focal_loss'
    ),
    'bn_inception_wgap_emb': ModelInfo(
        classification_model_bninception_wgap,
        pretrained_settings=pretrainedmodels.pretrained_settings['bninception']['imagenet'],
        args=dict(
            gwap_channels=(192, 576, 1056, 1024),
            fc_layers=(2848, 256),
            scale=3,
            dropout=0.0),
        batch_size=32,
        dataset_args=dict(geometry_aug_level=10, img_aug_level=20, use_extarnal=True),
        folds_split='emb',
        optimiser='sgd',
        initial_lr=0.1,
        loss='bce',
        nb_epochs=256
    ),
}
