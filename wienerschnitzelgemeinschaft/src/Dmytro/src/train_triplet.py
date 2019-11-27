import argparse
import collections
import os

import cv2
import numpy as np
import pandas as pd
import pretrainedmodels
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
from triplet_dataset import TripletDataset, TripletDatasetUpdate, TripletDatasetPredict
from logger import Logger
from experiments import MODELS

class EmbeddingsModel(nn.Module):
    def __init__(self, nb_embeddings=config.NB_EMBEDDINGS):
        super().__init__()
        self.base_model = pretrainedmodels.resnet18()
        self.fc = nn.Linear(2048, nb_embeddings)

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def train(model_name, run=None):
    run_str = '' if run is None or run == '' else f'_{run}'

    checkpoints_dir = f'../output/checkpoints_3/{model_name}{run_str}'
    tensorboard_dir = f'../output/tensorboard_3/{model_name}{run_str}'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    print('\n', model_name, '\n')

    logger = Logger(tensorboard_dir)

    model = EmbeddingsModel()
    model = model.cuda()

    dataset_train = TripletDataset(
        is_train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        crop_size=256
    )

    dataset_valid = TripletDataset(
        is_train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        crop_size=256
    )

    dataset_update_train = TripletDatasetUpdate(dataset_train)
    dataset_update_valid = TripletDatasetUpdate(dataset_valid)

    model.training = True

    print('using sgd optimiser')
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    scheduler = utils.CosineAnnealingLRWithRestarts(optimizer, T_max=8, T_mult=1.2)

    print('Num training images: {} valid images: {}'.format(len(dataset_train), len(dataset_valid)))

    data_loader_train = DataLoader(
        dataset_train,
        shuffle=True,
        num_workers=8,
        batch_size=64)

    data_loader_valid = DataLoader(
        dataset_valid,
        shuffle=False,
        num_workers=8,
        batch_size=64)

    data_loader_update_train = DataLoader(
        dataset_update_train,
        shuffle=False,
        num_workers=8,
        batch_size=64)

    data_loader_update_valid = DataLoader(
        dataset_update_valid,
        shuffle=False,
        num_workers=8,
        batch_size=64)

    criterium = TripletLoss(margin=1.0)

    for epoch_num in range(512):
        model.eval()
        with torch.set_grad_enabled(False):
            for iter_num, data in tqdm(enumerate(data_loader_update_train), total=len(data_loader_update_train)):
                img = data['img'].cuda()
                samples_idx = data['idx']

                vectors = model(img).detach().cpu().numpy()
                dataset_train.embeddings[samples_idx] = vectors
            print(np.mean(dataset_train.embeddings, axis=0), np.std(dataset_train.embeddings, axis=0))

            for iter_num, data in tqdm(enumerate(data_loader_update_valid), total=len(data_loader_update_valid)):
                img = data['img'].cuda()
                samples_idx = data['idx']

                vectors = model(img).detach().cpu().numpy()
                dataset_valid.embeddings[samples_idx] = vectors
            print(np.mean(dataset_train.embeddings, axis=0), np.std(dataset_valid.embeddings, axis=0))

        model.train()
        epoch_loss = []
        with torch.set_grad_enabled(True):
            data_iter = tqdm(enumerate(data_loader_train), total=len(data_loader_train))
            for iter_num, data in data_iter:
                img = data['img'].cuda()
                img_pos = data['img_pos'].cuda()
                img_neg = data['img_neg'].cuda()

                optimizer.zero_grad()
                output = model(img)
                output_pos = model(img_pos)
                output_neg = model(img_neg)

                loss = criterium(output, output_pos, output_neg)
                epoch_loss.append(float(loss.detach().cpu()))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()

                data_iter.set_description(
                    f'{epoch_num} Loss: {np.mean(epoch_loss):1.4f}')

        logger.scalar_summary(f'loss_train', np.mean(epoch_loss), epoch_num)

        epoch_loss = []
        with torch.set_grad_enabled(False):
            data_iter = tqdm(enumerate(data_loader_valid), total=len(data_loader_valid))
            for iter_num, data in data_iter:
                img = data['img'].cuda()
                img_pos = data['img_pos'].cuda()
                img_neg = data['img_neg'].cuda()

                output = model(img)
                output_pos = model(img_pos)
                output_neg = model(img_neg)

                loss = criterium(output, output_pos, output_neg)
                epoch_loss.append(float(loss))

                data_iter.set_description(
                    f'{epoch_num} Loss: {np.mean(epoch_loss):1.4f}')

        logger.scalar_summary(f'loss_valid', np.mean(epoch_loss), epoch_num)

        logger.scalar_summary('lr', optimizer.param_groups[0]['lr'], epoch_num)

        scheduler.step(metrics=np.mean(epoch_loss), epoch=epoch_num)

        model.eval()
        torch.save(
            {
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            f'{checkpoints_dir}/{epoch_num:03}.pt'
        )


def predict(model_name, epoch_num, img_dir, sample_ids, run=None):
    model = EmbeddingsModel()
    model = model.cuda()

    run_str = '' if run is None or run == '' else f'_{run}'
    checkpoints_dir = f'../output/checkpoints_3/{model_name}{run_str}'

    checkpoint = torch.load(f'{checkpoints_dir}/{epoch_num:03}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    dataset = TripletDatasetPredict(sample_ids=sample_ids,
                                    img_dir=img_dir,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]),
                                    crop_size=256)

    data_loader_update_train = DataLoader(
        dataset,
        shuffle=False,
        num_workers=8,
        batch_size=64)

    results = []
    results_idx = []
    with torch.set_grad_enabled(False):
        for data in tqdm(data_loader_update_train):
            img = data['img'].cuda()
            samples_idx = data['idx']
            embeddings = model(img).detach().cpu().numpy()
            results.append(embeddings)
            results_idx.append(samples_idx)

        # for image_id in tqdm(sample_ids):
        #     images = []
        #     for color in ['red', 'green', 'blue']:
        #         try:
        #             img = cv2.imread(f'{img_dir}/{image_id}_{color}.png', cv2.IMREAD_UNCHANGED)
        #             img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA).astype("uint8")
        #             images.append(img)
        #         except:
        #             print(f'failed to open {img_dir}/{image_id}_{color}.png')
        #             raise
        #
        #     images = np.stack(images, axis=0).astype(np.float32) / 255.0
        #     images = torch.from_numpy(images).cuda()
        #     images = normalize(images)
        #     images = torch.unsqueeze(images, 0)
        #     embeddings = model(images)
        #     embeddings = embeddings.detach().cpu().numpy()
        #     # print(image_id, embeddings.flatten())
        #     results.append(embeddings)

    results = np.concatenate(results, axis=0)
    results_idx = np.concatenate(results_idx, axis=0)
    utils.print_stats('results_idx diff', np.diff(results_idx))
    return results


def predict_extra(model_name, epoch_num, run=None):
    data = pd.read_csv('../input/folds_4_extra.csv')
    embeddings = predict(model_name=model_name, epoch_num=epoch_num,
                         img_dir=config.TRAIN_DIR_EXTRA,
                         sample_ids=data.Id.values,
                         run=run)
    torch.save(embeddings, '../output/embeddings_extra.pt')
    for i in range(embeddings.shape[1]):
        data[f'emb_{i}'] = embeddings[:, i]

    print(np.mean(embeddings, axis=0), np.std(embeddings, axis=0))

    data.to_csv('../input/emb_extra.csv', index=False)


def predict_train(model_name, epoch_num, run=None):
    data = pd.read_csv('../input/train.csv')
    embeddings = predict(model_name=model_name, epoch_num=epoch_num,
                         img_dir=config.TRAIN_DIR,
                         sample_ids=data.Id.values,
                         run=run)
    torch.save(embeddings, '../output/embeddings_train.pt')
    for i in range(embeddings.shape[1]):
        data[f'emb_{i}'] = embeddings[:, i]

    print(np.mean(embeddings, axis=0), np.std(embeddings, axis=0))

    data.to_csv('../input/emb_train.csv', index=False)


def predict_test(model_name, epoch_num, run=None):
    data = pd.read_csv('../input/sample_submission.csv')
    embeddings = predict(model_name=model_name, epoch_num=epoch_num,
                         img_dir=config.TEST_DIR,
                         sample_ids=data.Id.values,
                         run=run)
    torch.save(embeddings, '../output/embeddings_test.pt')
    for i in range(embeddings.shape[1]):
        data[f'emb_{i}'] = embeddings[:, i]

    print(np.mean(embeddings, axis=0), np.std(embeddings, axis=0))

    data.to_csv('../input/emb_test.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='train')
    parser.add_argument('--model', type=str, default='tr_resnet18_now_8')
    parser.add_argument('--epoch', type=int, default=0)

    args = parser.parse_args()
    action = args.action
    model = args.model

    np.set_printoptions(precision=3, linewidth=200)

    if action == 'train':
        try:
            train(model_name=model)
        except KeyboardInterrupt:
            pass

    if action == 'predict_extra':
        predict_extra(model_name=model, epoch_num=args.epoch, run=None)

    if action == 'predict_train':
        predict_train(model_name=model, epoch_num=args.epoch, run=None)

    if action == 'predict_test':
        predict_test(model_name=model, epoch_num=args.epoch, run=None)
