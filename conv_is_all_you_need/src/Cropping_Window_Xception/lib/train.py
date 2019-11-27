# TODO: remove the large classes in the external data (to reduce labeling noise)
# TODO: Multi-label stratification
# TODO: make the printing better looking.

# TODO: URGENT: should *not* use the number of windows, as the total number of windows change
# TODO: PICKUP: setup a robust CV framework

# TODO: Many difficult classes can get their *own* specialized classifier

# TODO: IDEA: visualize the heatmap of the map's attention
# TODO: IDEA: with test-time augmentation, it's interesting to see if it's frequent that the augmentations got different predictions (good way to evaluate the robustness of the model)

# TODO: IDEA: add more inception units in the network to compensate for the size increase.
# TODO: PICKUP: threshold picking is a big problem for the current model
# TODO: IDEA: use multiple annihilation to increase the stability of the model
# TODO: SQUEEZE: use all training example without splitting validation
# TODO: SQUEEZE: find more identical pairs between the HPA data and the test set. use per-channel correlation
# TODO: SQUEEZE: 9 (endosomes) and 10 (lysosomes) have a *solid* 1 correlation, and negative correlation with 26 (cytoplasmic bodies), consider manually binding them together

# TODO: SQUEEZE: a better augmentation might help for some rare classesdd

# 1) pretraining
# 2) single-label samples
# 3) no early stopping (or any validation-based stopping)
# 4) low-res first

import json
import copy
import os
from os.path import join as pjoin

import GPUtil
import pandas as pd
import numpy as np

from .config import print_config
from .ignite_trainer import train
# from .train_with_keras import train
from .utils import chunk, debug, gen_cwd_slash, multiprocessing, compute_i_coords, info
import math


def train_task_generator(config, i_begin, i_end, avail_gpus):
    debug(f"config['class_ids'] = {config['class_ids']}")
    for i_fold in range(config['n_folds']):
        config_for_one_class = copy.deepcopy(config)
        config_for_one_class.update(
            {
                '_cwd': pjoin(config['_cwd'], f'fold_{i_fold}'),
                'cuda_visible_devices': str(avail_gpus[i_fold % len(avail_gpus)]),
                'verbose': 1,
                'i_fold': i_fold,
                'path_to_train_anno_cache': pjoin(config['_cwd'], f'fold_{i_fold}', 'train_anno.csv'),
                'path_to_train_windowed_anno_cache': pjoin(config['_cwd'], f'fold_{i_fold}', 'train_windowed_anno.csv'),
                'path_to_valid_anno_cache': pjoin(config['_cwd'], f'fold_{i_fold}', 'valid_anno.csv'),
                'path_to_valid_windowed_anno_cache': pjoin(config['_cwd'], f'fold_{i_fold}', 'valid_windowed_anno.csv'),
            }
        )
        yield config_for_one_class


def train_all(config):
    cwd_slash = gen_cwd_slash(config)

    os.makedirs(config['_cwd'], exist_ok=True)

    if not os.path.isfile(cwd_slash('config.json')):
        with open(cwd_slash('config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    if not os.path.isfile(config['path_to_valid_windowed_anno_cache']):
        split_train_valid(config)

    train(config)

    return {'id_': config['i_fold']}


def train_folds(config):
    os.makedirs(config['_cwd'], exist_ok=True)

    cwd_slash = gen_cwd_slash(config)

    if not os.path.isfile(cwd_slash('config.json')):
        with open(cwd_slash('config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    if not os.path.isdir(cwd_slash('fold_0')):
        split_into_folds(config)

    avail_gpus = GPUtil.getAvailable(limit=100)
    debug(f"avail_gpus = {avail_gpus}")

    for i_begin, i_end in chunk(config['n_folds'], 28):
        list(
            multiprocessing(
                train_all,
                train_task_generator(config, i_begin, i_end, avail_gpus),
                len_=config['n_folds'],
                n_threads=None,
                disabled=False,
            )
        )


def split_into_folds(config):
    info('split_into_folds()')
    cwd_slash = gen_cwd_slash(config)

    anno = pd.read_csv(config['path_to_train_anno'], index_col=0)

    if config['subsampling'] is not None:
        anno = anno.sample(config['subsampling'], random_state=config['_random_state'])
        # anno = anno.tail(config['subsampling'])

    idxs = np.random.permutation(anno.index.values)
    folds_idxs = [idxs[a:b] for a, b in chunk(len(idxs), math.ceil(len(idxs) / config['n_folds']))]

    fold_annos = [anno.loc[fold_idxs] for fold_idxs in folds_idxs]

    for i_fold in range(len(fold_annos)):
        os.makedirs(cwd_slash(f"fold_{i_fold}"))
        train_list = fold_annos.copy()
        valid_anno = train_list.pop(i_fold)
        valid_anno.to_csv(cwd_slash(f"fold_{i_fold}", f"valid_anno.csv"))
        train_anno = pd.concat(train_list)
        train_anno.to_csv(cwd_slash(f"fold_{i_fold}", f"train_anno.csv"))

    # n_valid_samples = round(len(anno) * config['frac_of_validation_samples'])
    # anno = anno.sample(len(anno), random_state=config['_random_state'])
    # train_anno = anno.head(len(anno) - n_valid_samples)
    # train_anno.to_csv(train_anno_path)
    # valid_anno = anno.tail(n_valid_samples)
    # valid_anno.to_csv(valid_anno_path)


def split_train_valid(config):
    info('split_train_valid()')
    anno = pd.read_csv(config['path_to_train_anno'], index_col=0)

    if config['subsampling'] is not None:
        anno = anno.sample(config['subsampling'], random_state=config['_random_state'])
        # anno = anno.tail(config['subsampling'])

    train_anno_path = config['path_to_train_anno_cache']
    valid_anno_path = config['path_to_valid_anno_cache']
    train_windowed_anno_path = config['path_to_train_windowed_anno_cache']
    valid_windowed_anno_path = config['path_to_valid_windowed_anno_cache']

    # train/valid split
    if os.path.isfile(train_anno_path) and os.path.isfile(valid_anno_path):
        debug(f'using train/valid cache from {train_anno_path} and {valid_anno_path}')
        train_anno = pd.read_csv(train_anno_path, index_col=0)
        valid_anno = pd.read_csv(valid_anno_path, index_col=0)
    else:
        n_valid_samples = round(len(anno) * config['frac_of_validation_samples'])
        anno = anno.sample(len(anno), random_state=config['_random_state'])
        train_anno = anno.head(len(anno) - n_valid_samples)
        train_anno.to_csv(train_anno_path)
        valid_anno = anno.tail(n_valid_samples)
        valid_anno.to_csv(valid_anno_path)

    train_windowed_anno = pd.read_csv(config['path_to_train_windowed_anno'], index_col=0)
    train_windowed_anno = compute_i_coords(train_windowed_anno, config)
    train_windowed_anno = train_windowed_anno.join(train_anno[['group']], how='right', on='source_img_id')
    train_windowed_anno.to_csv(train_windowed_anno_path)
    valid_windowed_anno = pd.read_csv(config['path_to_valid_windowed_anno'], index_col=0)
    valid_windowed_anno = compute_i_coords(valid_windowed_anno, config)
    valid_windowed_anno = valid_windowed_anno.join(valid_anno[['group']], how='right', on='source_img_id')
    valid_windowed_anno.to_csv(valid_windowed_anno_path)

    debug(f'len(train_windowed_anno) = {len(train_windowed_anno)}')
    debug(f'len(valid_windowed_anno) = {len(valid_windowed_anno)}')
