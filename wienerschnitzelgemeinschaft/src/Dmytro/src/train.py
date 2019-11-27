import argparse
import collections
import os

import numpy as np
import torch
import torch.optim as optim
import torchsummary
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import skimage.io
from sklearn.metrics import f1_score

import torch.nn as nn
import torch.nn.functional as F
import config
import utils
import classification_dataset
from classification_dataset import ClassificationDataset
from logger import Logger

from config import NB_CATEGORIES
from experiments import MODELS


class SemiBalancedSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: ClassificationDataset, epoch_size, balanced_sampling_ratio=0.5):
        super().__init__(data_source)
        self.balanced_sampling_ratio = balanced_sampling_ratio
        self.data_source = data_source
        self.epoch_size = epoch_size

    def generator(self):
        # np.random.seed()
        for i in range(self.epoch_size):
            selection_policy = np.random.random()
            if selection_policy > self.balanced_sampling_ratio:
                yield np.random.randint(0, len(self.data_source))
            else:
                feature = np.random.randint(0, config.NB_CATEGORIES)
                yield np.random.choice(self.data_source.samples_idx_by_label[feature])

    def __iter__(self):
        return self.generator()

    def __len__(self):
        return self.epoch_size


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, target):
        if not (target.size() == inputs.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), inputs.size()))

        max_val = (-inputs).clamp(min=0)
        loss = inputs - inputs * target + max_val + ((-max_val).exp() + (-inputs - max_val).exp()).log()

        invprobs = F.logsigmoid(-inputs * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


class FocalLoss2(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, target):
        if not (target.size() == inputs.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), inputs.size()))

        bce_loss = F.binary_cross_entropy_with_logits(inputs, target, reduction='none')
        pt = torch.exp(-bce_loss)
        f_loss = (1-pt)**self.gamma * bce_loss

        return torch.mean(f_loss) * 28


class F1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.exp(y_pred)

        tp = torch.sum(y_true * y_pred, dim=0)
        # tn = torch.sum((1 - y_true) * (1 - y_pred), axis=0)
        fp = torch.sum((1 - y_true) * y_pred, dim=0)
        fn = torch.sum(y_true * (1 - y_pred), dim=0)

        eps = 1e-6

        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)

        f1 = 2 * p * r / (p + r + eps)

        f1[torch.isnan(f1)] = 0
        return 1 - torch.mean(f1)


def balanced_sampling_ratio(epoch):
    return np.clip(0.5 - epoch*0.025, 0.05, 1.0)


def train(model_name, fold, run=None, resume_epoch=-1):
    run_str = '' if run is None else f'_{run}'

    model_info = MODELS[model_name]

    checkpoints_dir = f'../output/checkpoints/{model_name}{run_str}_{fold}'
    tensorboard_dir = f'../output/tensorboard/{model_name}{run_str}_{fold}'
    oof_dir = f'../output/oof/{model_name}{run_str}_{fold}'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(oof_dir, exist_ok=True)
    print('\n', model_name, '\n')

    logger = Logger(tensorboard_dir)

    model = model_info.factory(**model_info.args)
    model = model.cuda()
    # try:
    #     torchsummary.summary(model, (4, 512, 512))
    #     print('\n', model_name, '\n')
    # except:
    #     raise
    #     pass

    model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    dataset_train = ClassificationDataset(
        fold=fold,
        is_training=True,
        transform=lambda x: torch.from_numpy(x),
        crop_size=model_info.crop_size,
        folds_split=model_info.folds_split,
        **model_info.dataset_args
    )

    dataset_valid = ClassificationDataset(
        fold=fold,
        is_training=False,
        transform=lambda x: torch.from_numpy(x),
        crop_size=model_info.crop_size,
        folds_split=model_info.folds_split,
        # use_extarnal=model_info.dataset_args.get('use_extarnal', False)
    )

    epoch_size = 20000

    model.training = True

    if model_info.optimiser == 'adam':
        print('using adam optimiser')
        optimizer = optim.Adam(model.parameters(), lr=model_info.initial_lr)
    else:
        print('using sgd optimiser')
        optimizer = optim.SGD(model.parameters(), lr=model_info.initial_lr, momentum=0.9, weight_decay=1e-5)

    if model_info.scheduler == 'cos':
        scheduler = utils.CosineAnnealingLRWithRestarts(optimizer, T_max=8, T_mult=1.2)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=16, verbose=True, factor=0.2)

    print('Num training images: {}'.format(len(dataset_train)))

    if resume_epoch > -1:
        checkpoint = torch.load(f'{checkpoints_dir}/{resume_epoch:03}.pt')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    data_loaders = {
        'train': DataLoader(dataset_train,
                            num_workers=16,
                            batch_size=model_info.batch_size,
                            sampler=SemiBalancedSampler(dataset_train, epoch_size)),
        'val':   DataLoader(dataset_valid,
                            num_workers=16,
                            batch_size=8,
                            drop_last=True)
    }

    loss_scale = {}

    if model_info.loss == 'bce':
        print('using bce loss')
        criterium = {
            'bce': nn.BCEWithLogitsLoss()
        }
        loss_scale = {
            'bce': 10.0
        }
    elif model_info.loss == 'focal_loss2':
        print('using focal loss2')
        criterium = {
            'fl': FocalLoss2()
        }
    elif model_info.loss == 'focal_loss':
        print('using focal loss')
        criterium = {
            'fl': FocalLoss()
        }
    elif model_info.loss == 'bce_f1':
        print('using focal loss')
        criterium = {
            'f1': F1Loss(),
            'bce': nn.BCEWithLogitsLoss()
        }

        loss_scale = {
            'f1': 1.0,
            'bce': 10.0
        }

    for epoch_num in range(resume_epoch+1, model_info.nb_epochs):
        # scheduler.step(epoch=epoch_num)
        data_loaders['train'].sampler.balanced_sampling_ratio = balanced_sampling_ratio(epoch_num)

        for phase in ['train', 'val']:
            epoch_results_true = []
            epoch_results_output = []

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            if model_info.is_pretrained:
                if epoch_num < 1:
                    model.module.freeze_encoder()
                elif epoch_num == 1:
                    model.module.unfreeze_encoder()

            epoch_loss_items = collections.defaultdict(list)
            epoch_loss = []

            data_loader = data_loaders[phase]
            data_iter = tqdm(enumerate(data_loader), total=len(data_loader))
            for iter_num, data in data_iter:
                img = data['img'].cuda()
                labels = data['labels'].cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(img)

                    total_loss = []
                    for loss_name, loss_fn in criterium.items():
                        loss = loss_fn(output, labels)
                        epoch_loss_items[loss_name].append(float(loss))
                        total_loss.append(loss * loss_scale.get(loss_name, 1.0))

                    epoch_loss.append(float(sum(total_loss)))

                    if phase == 'train':
                        sum(total_loss).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                        optimizer.step()
                    epoch_results_true.append(data['labels'].detach().cpu().numpy())
                    epoch_results_output.append(torch.sigmoid(output).detach().cpu().numpy())

                data_iter.set_description(
                    f'{epoch_num} Loss: {np.mean(epoch_loss):1.4f}')

            for loss_name, epoch_loss_item in epoch_loss_items.items():
                logger.scalar_summary(f'loss_{loss_name}_{phase}', np.mean(epoch_loss_item), epoch_num)
            logger.scalar_summary(f'loss_{phase}', np.mean(epoch_loss), epoch_num)

            logger.scalar_summary('loss_'+phase, np.mean(epoch_loss), epoch_num)
            logger.scalar_summary('lr', optimizer.param_groups[0]['lr'], epoch_num) # scheduler.get_lr()[0]

            epoch_results_true = np.concatenate(epoch_results_true, axis=0)
            epoch_results_output = np.concatenate(epoch_results_output, axis=0)
            score = np.mean([f1_score(epoch_results_true[:, i],
                                      epoch_results_output[:, i] > 0.375,
                                      average='binary')
                             for i in range(config.NB_CATEGORIES)])

            logger.scalar_summary('f1_' + phase, score, epoch_num)

            if phase == 'val':
                scheduler.step(metrics=np.mean(epoch_loss), epoch=epoch_num)
                np.save(f'{oof_dir}/{epoch_num:03}.npy', np.array([epoch_results_true, epoch_results_output]))

            if epoch_num % 2 == 0:
                torch.save(
                    {
                        'epoch': epoch_num,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    },
                    f'{checkpoints_dir}/{epoch_num:03}.pt'
                )

    model.eval()
    torch.save(model.state_dict(), f'{checkpoints_dir}/{model_name}_final.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='check')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--run', type=str, default='')
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--epoch', type=int, default=-1)

    parser.add_argument('--resume_weights', type=str, default='')
    parser.add_argument('--resume_epoch', type=int, default=-1)

    args = parser.parse_args()
    action = args.action
    model = args.model
    fold = args.fold

    if action == 'train':
        try:
            train(model_name=model, run=args.run, fold=args.fold, resume_epoch=args.resume_epoch)
        except KeyboardInterrupt:
            pass
