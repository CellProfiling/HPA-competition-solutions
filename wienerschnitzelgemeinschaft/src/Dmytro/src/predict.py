import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchsummary
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from typing import List
import skimage.io
from sklearn.metrics import f1_score

import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels

import config
import matplotlib.pyplot as plt

import classification_dataset
from classification_dataset import ClassificationDataset,ClassificationDatasetTest
from logger import Logger

from config import NB_CATEGORIES
from experiments import MODELS

import yaml

train_probs = {
    0: 41.47,
    1: 4.04,
    2: 11.65,
    3: 5.02,
    4: 5.98,
    5: 8.09,
    6: 3.24,
    7: 9.08,
    8: 0.17,
    9: 0.14,
    10: 0.09,
    11: 3.52,
    12: 2.21,
    13: 1.73,
    14: 3.43,
    15: 0.07,
    16: 1.71,
    17: 0.68,
    18: 2.90,
    19: 4.77,
    20: 0.55,
    21: 12.16,
    22: 2.58,
    23: 9.54,
    24: 1.04,
    25: 26.48,
    26: 1.06,
    27: 0.04,
}


def tta_images(img):
    rotated = img.transpose(2, 3)
    res = [img,
           img.flip(2),
           img.flip(3),
           img.flip(2, 3),
           rotated,
           rotated.flip(2),
           rotated.flip(3),
           rotated.flip(2, 3)]

    return res


def test_tta_images():
    plane = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 1, 1]], dtype=np.uint8)
    img = np.array([[plane]])
    print(img.shape)
    for t in tta_images(torch.tensor(img)):
        print(t)

# test_tta_images()


def fit_prob_th(results, scale_small=1.0):
    thresholds = []
    for cat, train_prob in train_probs.items():
        train_prob /= 100.0
        target_prob = (train_prob**scale_small) / (0.5 ** scale_small) / 1.95
        target_count = int(results.shape[0] * target_prob)
        target_count = np.clip(target_count, 0, results.shape[0]-1)
        threshold = np.sort(results[:, cat].copy())[-target_count]
        thresholds.append(threshold)

        print(f'{cat:02}  {threshold:0.3f} {target_prob:0.3f} {target_count}')
    return np.array(thresholds)


def predict(submission_name):
    results = []

    dataset = ClassificationDatasetTest(transform=lambda x: torch.from_numpy(x))
    data_loader = DataLoader(
        dataset,
        num_workers=8,
        batch_size=4,
        drop_last=False)

    submission_config = yaml.load(open(f'../submissions/{submission_name}.yaml'))
    os.makedirs('../output/predict_cache', exist_ok=True)
    use_tta = submission_config.get('use_tta', False)

    for model_cfg in submission_config['models']:
        model_name = model_cfg['model']
        run = model_cfg.get('run', '')
        folds = model_cfg.get('folds', config.FOLDS)
        run_str = '' if run is None else f'_{run}'
        model_info = MODELS[model_name]
        print(model_name)

        for fold in folds:
            try:
                checkpoint = model_cfg['fold_checkpoints'][fold]
            except IndexError:
                continue
            except KeyError:
                continue

            checkpoints_dir = f'../output/checkpoints/{model_name}{run_str}_{fold}'
            print(f'fold {fold} checkpoint {checkpoint}')

            suffix = '_tta' if use_tta else ''
            cache_fn = f'../output/predict_cache/{model_name}{run_str}_fold_{fold}_ch_{checkpoint:03}{suffix}.pt'

            try:
                fold_outputs = torch.load(cache_fn)
            except FileNotFoundError:
                model = model_info.factory(**model_info.args)
                state_dict = torch.load(f'{checkpoints_dir}/{checkpoint:03}.pt')
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                model.load_state_dict(state_dict)
                model = model.cuda()
                model.eval()

                with torch.set_grad_enabled(False):
                    fold_outputs = []
                    for iter_num, data in tqdm(enumerate(data_loader), total=len(data_loader)):
                        tta_predictions = []
                        images = tta_images(data['img']) if use_tta else [data['img']]
                        for img in images:
                            output = model(img.cuda())
                            output = torch.sigmoid(output)
                            tta_predictions.append(output.detach().cpu().numpy())
                        fold_outputs.append(np.stack(tta_predictions, axis=1))
                    fold_outputs = np.concatenate(fold_outputs, axis=0)
                    torch.save(fold_outputs, cache_fn)
            results.append(fold_outputs)  # dimensions: fold, id, tta, class_predictions
    results = np.array(results)
    print(results.shape)
    results = np.mean(results, axis=(0, 2))

    if 'thresholds' in submission_config:
        if submission_config['thresholds'] == 'fix_prob':
            threshold = fit_prob_th(results, scale_small=submission_config['thresholds_scale'])
        else:
            threshold = np.array(submission_config['thresholds'])
    else:
        threshold = 0.5

    pred_list = []
    for line in results:
        predictions = np.nonzero(line > threshold)[0]
        s = ' '.join(list([str(i) for i in predictions]))
        # if len(predictions) == 0:
        #     s = str(np.argmax(line / threshold))
        pred_list.append(s)

    sample_df = dataset.data
    sample_df.Predicted = pred_list
    sample_df.to_csv(f'../submissions/{submission_name}.csv', header=True, index=False)


def print_perc_and_f1_from_prob(data, labels, thresholds_sub, threshold=0.375):
    # samples = defaultdict(int)
    # for line in data:
    #     for i in np.nonzero(line > threshold)[0]:
    #         samples[i] += 1

    f1_hist = []
    f1_05_hist = []
    f1_opt_hist = []
    f1_sub_hist = []

    for key in range(config.NB_CATEGORIES):
        f1 = f1_score(labels[:, key], data[:, key] > threshold, average='binary')
        f1_05 = f1_score(labels[:, key], data[:, key] > threshold, average='binary')
        f1_opt = f1_score(labels[:, key], data[:, key] > f1/2+1e-3, average='binary')
        f1_sub = f1_score(labels[:, key], data[:, key] > thresholds_sub[key], average='binary')
        prob_sub = np.sum(data[:, key] > thresholds_sub[key]) * 100.0 / data.shape[0]
        prob_05 = np.sum(data[:, key] > threshold) * 100.0 / data.shape[0]

        f1_hist.append(f1)
        f1_05_hist.append(f1_05)
        f1_opt_hist.append(f1_opt)
        f1_sub_hist.append(f1_sub)

        # prob = samples[key] * 100.0 / data.shape[0]
        print(f'{key:3}  train {train_probs[key]:02.2f}  0.5: {prob_05:02.2f}  sub: {prob_sub:02.2f} '
              f'f1 th {threshold} {f1:0.2f}  '
              f'f1 th {f1/2+1e-3:0.2f} {f1_opt:0.2f}  '
              f'f1 sub th {thresholds_sub[key]} {f1_sub:0.2f}')

    print(f'F1 0.5 {np.mean(f1_05_hist):0.2}   {threshold} {np.mean(f1_hist):0.2} th f1/2 {np.mean(f1_opt_hist):0.2} '
          f'th sub {np.mean(f1_sub_hist):0.2}')


def predict_oof(submission_name):
    results = defaultdict(list)

    submission_config = yaml.load(open(f'../submissions/{submission_name}.yaml'))

    # make sure all models using the same folds split
    model_name = submission_config['models'][0]['model']
    model_info = MODELS[model_name]

    datasets = [
        ClassificationDataset(
            fold=fold,
            is_training=False,
            transform=lambda x: torch.from_numpy(x),
            folds_split=model_info.folds_split
        )
        for fold in config.FOLDS
    ]

    data_loaders = [
        DataLoader(
            datasets[fold],
            num_workers=8,
            batch_size=16,
            drop_last=False)
        for fold in config.FOLDS
    ]

    os.makedirs('../output/predict_oof', exist_ok=True)

    all_fold_labels = {}

    for model_cfg in submission_config['models']:
        model_name = model_cfg['model']
        run = model_cfg.get('run', '')
        folds = model_cfg.get('folds', config.FOLDS)
        run_str = '' if run is None else f'_{run}'
        model_info = MODELS[model_name]
        print(model_name)

        for fold in folds:
            checkpoint = model_cfg['fold_checkpoints'][fold]
            checkpoints_dir = f'../output/checkpoints/{model_name}{run_str}_{fold}'
            print(f'fold {fold} checkpoint {checkpoint}')

            cache_fn = f'../output/predict_oof/{model_name}{run_str}_fold_{fold}_ch_{checkpoint:03}.pt'

            try:
                fold_outputs, fold_labels = torch.load(cache_fn)
            except FileNotFoundError:
                model = model_info.factory(**model_info.args)
                state_dict = torch.load(f'{checkpoints_dir}/{checkpoint:03}.pt')
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                model.load_state_dict(state_dict)
                model = model.cuda()
                model.eval()

                with torch.set_grad_enabled(False):
                    fold_outputs = []
                    fold_labels = []
                    for iter_num, data in tqdm(enumerate(data_loaders[fold]), total=len(data_loaders[fold])):
                        img = data['img'].cuda()
                        labels = data['labels'].detach().cpu().numpy()
                        output = model(img)
                        output = torch.sigmoid(output)
                        fold_labels.append(labels)
                        fold_outputs.append(output.detach().cpu().numpy())

                    fold_labels = np.concatenate(fold_labels, axis=0)
                    fold_outputs = np.concatenate(fold_outputs, axis=0)
                    torch.save((fold_outputs, fold_labels), cache_fn)
                    all_fold_labels[fold] = fold_labels

            print_perc_and_f1_from_prob(fold_outputs, fold_labels, thresholds_sub=submission_config['thresholds'])
            results[fold].append(fold_outputs)

    for fold in results.keys():
        fold_results = np.mean(np.array(results[fold]), axis=0)
        print('fold', fold)
        print_perc_and_f1_from_prob(fold_results, all_fold_labels[fold], thresholds_sub=submission_config['thresholds'])


def find_threshold(data, cls, plot=None, plot_label=''):
    # threshold_table = np.zeros((data.shape[0], 2))
    # threshold_table[:, 0] = sorted(data[:, cls])
    # for i, th in enumerate(threshold_table[:, 0]):
    #     f1 = f1_score(gt, data[:, cls] > th, average='macro')
    #     threshold_table[i, 1] = f1
    gt = data[1][:, cls]
    thresholds = []
    f1_values = []
    th = 1e-3
    while th < 1:
        f1 = f1_score(gt, data[0][:, cls] > th, average='binary')
        thresholds.append(th)
        f1_values.append(f1)
        th *= 1.025

    top_threshold = thresholds[np.argmax(f1_values)]

    print(f'cls {cls} th {top_threshold:.03} f1 {max(f1_values):.3}')
    if plot:
        plot.plot(thresholds, f1_values, label=plot_label)
    return thresholds, f1_values


def find_threshold_from_oof(submission_name):
    results = defaultdict(list)

    submission_config = yaml.load(open(f'../submissions/{submission_name}.yaml'))

    # make sure all models using the same folds split
    model_name = submission_config['models'][0]['model']
    model_info = MODELS[model_name]

    datasets = [
        ClassificationDataset(
            fold=fold,
            is_training=False,
            transform=lambda x: torch.from_numpy(x),
            folds_split=model_info.folds_split
        )
        for fold in config.FOLDS
    ]

    submission_config = yaml.load(open(f'../submissions/{submission_name}.yaml'))
    for model_cfg in submission_config['models']:
        model_name = model_cfg['model']
        run = model_cfg.get('run', '')
        folds = model_cfg.get('folds', config.FOLDS)
        run_str = '' if run is None else f'_{run}'
        print(model_name)
        model_fold_outputs = []

        for fold in folds:
            checkpoint = model_cfg['fold_checkpoints'][fold]
            print(f'fold {fold} checkpoint {checkpoint}')

            cache_fn = f'../output/predict_oof/{model_name}{run_str}_fold_{fold}_ch_{checkpoint:03}.pt'
            fold_outputs = torch.load(cache_fn)

            # print_perc_and_f1_from_prob(fold_outputs, thresholds_sub=submission_config['thresholds'])
            results[fold].append(fold_outputs)
            model_fold_outputs.append(fold_outputs)

        all_combined_f1 = []
        f, axx = plt.subplots(6, 5)
        for cls in range(config.NB_CATEGORIES):
            # plt.figure()
            ax = axx[cls // 5, cls % 5]
            ax.set_title(str(cls))
            print()
            thresholds = []
            f1_values = []
            for fold in folds:
                # samples = datasets[fold].samples
                # gt = np.array([sample.labels[cls] for sample in samples])

                fold_thresholds, fold_f1_values = find_threshold(
                    model_fold_outputs[fold],
                    cls=cls,
                    plot=ax,
                    plot_label=f'cls {cls} fold {fold}')
                thresholds.append(fold_thresholds)
                f1_values.append(fold_f1_values)

            thresholds = np.mean(np.array(thresholds), axis=0)
            f1_values = np.mean(np.array(f1_values), axis=0)
            top_threshold = thresholds[np.argmax(f1_values)]
            print(f'cls {cls} th {top_threshold:.03} f1 {max(f1_values):.3} combined')

            all_combined_f1.append(max(f1_values))

            # f1 = f1_score(gt, model_fold_outputs[fold][0][:, cls] > th, average='binary')
            # plt.legend()
        print('F1 ', np.mean(all_combined_f1))
        plt.show()

    # for fold in results.keys():
    #     fold_results = np.mean(np.array(results[fold]), axis=0)
    #     print('fold', fold)
    #     print_perc_and_f1_from_prob(fold_results, thresholds_sub=submission_config['thresholds'])



def check(submission_name):
    try:
        df = pd.read_csv(submission_name, dtype={'Id': str, 'Predicted': str}, na_values='')
        submission_config = yaml.load(submission_name[:-3]+'yaml')
    except FileNotFoundError:
        df = pd.read_csv(f'../submissions/{submission_name}.csv', dtype={'Id': str, 'Predicted': str}, na_values='')
        submission_config = yaml.load(open(f'../submissions/{submission_name}.yaml'))

    samples = defaultdict(int)
    for line in df.Predicted:
        line = str(line)
        if line == 'nan':
            continue
        for item in line.split():
            samples[int(item)] += 1

    for key in sorted(samples.keys()):
        prob = samples[key] * 100.0 / len(df.Predicted)
        threshold = submission_config['thresholds'][key]
        print(f'{key:3}  {prob:0.2f}  {train_probs[key]:0.2f}  {threshold}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='predict')
    parser.add_argument('--submission', type=str)

    args = parser.parse_args()
    action = args.action
    submission_name = args.submission

    if action == 'predict':
        predict(submission_name)
        check(submission_name)

    if action == 'check':
        check(submission_name)

    if action == 'predict_oof':
        predict_oof(submission_name)

    if action == 'find_threshold_from_oof':
        find_threshold_from_oof(submission_name)


if __name__ == '__main__':
    main()
