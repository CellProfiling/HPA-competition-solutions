import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
from tqdm import tqdm

from config.config import *
from utils.common_util import *
from utils.leak_util import *
from utils.ensemble_util import *

import argparse
parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--fold', default=0, type=int, help='index of fold (default: 0)')
parser.add_argument('--epoch_name', default='final', type=str, help='epoch name (default: final)')
parser.add_argument('--model_name', default='external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds',
                    type=str, help='complete model name')
parser.add_argument('--augments', default=None, type=str, help='')
parser.add_argument('--do_valid', default=1, type=int)
parser.add_argument('--do_test', default=1, type=int)
parser.add_argument('--update', default=0, type=int)
parser.add_argument('--seeds', default=None, type=str)
parser.add_argument('--ensemble_type', default='maximum', type=str, help='[maximum, average]')

def main():
    args = parser.parse_args()

    fold = args.fold
    fold = 'fold%d' % fold # fold0
    epoch_name = args.epoch_name
    epoch_name = 'epoch_%s' % epoch_name # epoch_final
    model_name = args.model_name
    augments = args.augments
    do_valid = args.do_valid == 1
    do_test = args.do_test == 1
    update = args.update == 1
    seeds = args.seeds
    ensemble_type = args.ensemble_type
    assert ensemble_type in ['maximum', 'average']

    if augments is None:
        augments = 'default'
    augments = augments.split(',')
    if seeds is not None:
        seeds = ['seed%s'%i for i in seeds.split(',')]
        augments = ['%s_%s'%(i, j) for i in augments for j in seeds]
    print(augments)

    train_df = pd.read_csv(opj(DATA_DIR, 'meta', 'train_meta.csv'))
    train_ratio = train_df[LABEL_NAME_LIST].mean().values
    # public lb ratio
    # https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/68678
    test_ratio = np.array([
        0.362397820,0.043841336,0.075268817,0.059322034,0.075268817,
        0.075268817,0.043841336,0.075268817,0.010000000,0.010000000,
        0.010000000,0.043841336,0.043841336,0.014198783,0.043841336,
        0.010000000,0.028806584,0.014198783,0.028806584,0.059322034,
        0.010000000,0.126126126,0.028806584,0.075268817,0.010000000,
        0.222493880,0.028806584,0.010000000
    ])
    idxes = np.where(test_ratio == 0.01)[0]
    test_ratio[idxes] = train_ratio[idxes]

    sub_dir = opj(RESULT_DIR, 'submissions', model_name, fold, epoch_name)
    to_dir = opj(RESULT_DIR, 'submissions', model_name, fold, epoch_name, 'whole_%s' % ensemble_type)
    os.makedirs(to_dir, exist_ok=True)

    threshold = 0.5
    kaggle_score = 0
    info_dfs = [pd.DataFrame(data=LABEL_NAME_LIST, columns=['label_name'])]
    if do_valid:
        data_type = 'val'
        df, probs = load_data(sub_dir, ensemble_type, data_type=data_type, augments=augments, update=update)
        threshold, kaggle_score, val_info_df = ensemble_augments(df, probs, to_dir, train_df, test_ratio, data_type=data_type,
                                                                 threshold=threshold, kaggle_score=kaggle_score, update=update)
        info_dfs.append(val_info_df)
    if do_test:
        data_type = 'test'
        df, probs = load_data(sub_dir, ensemble_type, data_type=data_type, augments=augments, update=update)
        threshold, kaggle_score, test_info_df = ensemble_augments(df, probs, to_dir, train_df, test_ratio, data_type=data_type,
                                                                  threshold=threshold, kaggle_score=kaggle_score, update=update)
        info_dfs.append(test_info_df)
    info_df = pd.concat(info_dfs, axis=1)
    for col in info_df.columns:
        if 'num' in col:
            info_df[col] = info_df[col].fillna(0).astype(int)

    print(info_df)
    info_df.to_csv(opj(to_dir, 'record_info.csv'), index=False)

def load_data(result_dir, ensemble_type, data_type='val', augments=None, threshold=0.5, update=False):
    out_dir = opj(result_dir, ensemble_type)
    os.makedirs(out_dir, exist_ok=True)

    prob_name = 'prob_%s.npy' % data_type
    result_name = 'results_%s.csv.gz' % data_type

    result_fname = opj(out_dir, result_name)
    prob_fname = opj(out_dir, prob_name)

    if ope(result_fname) and ope(prob_fname) and not update:
        result_df = pd.read_csv(result_fname)
        pred_probs = np.load(prob_fname)
    else:
        if augments is None:
            augments = ['default']

        result_df = None
        pred_probs = None
        for augment in augments:
            df = pd.read_csv(opj(result_dir, augment, result_name))
            probs = np.load(opj(result_dir, augment, prob_name))

            if result_df is None:
                result_df = df[[ID]]
                pred_probs = probs
            else:
                assert np.all(result_df[ID].values == df[ID].values)
                if ensemble_type == 'average':
                    pred_probs = pred_probs + probs
                else:
                    pred_probs = np.max(np.stack([pred_probs, probs], axis=-1), axis=-1)
        if ensemble_type == 'average':
            pred_probs = pred_probs / len(augments)

        labels = generate_labels(pred_probs, threshold=threshold)
        result_df[PREDICTED] = labels

        result_df.to_csv(result_fname, index=False, compression='gzip')
        np.save(prob_fname, pred_probs)

    return result_df, pred_probs

def merge_augments(to_dir, data_type, result_df, probs, update=False):
    whole_ids = result_df[ID].values

    prob_fname = opj(to_dir, 'prob_%s.npy' % data_type)
    if ope(prob_fname) and not update:
        whole_probs = np.load(prob_fname)
    else:
        whole_probs = probs
        np.save(prob_fname, whole_probs)
    return whole_ids, whole_probs

def ensemble_augments(result_df, probs, to_dir, train_df, data_ratio, data_type='val', threshold=0.5, kaggle_score=0, update=False):
    print('process %s dataset...' % data_type)
    result_df = result_df.copy()
    record_info = {}

    whole_ids, whole_probs = merge_augments(to_dir, data_type, result_df, probs, update=update)

    # search threshold
    if data_type == 'test':
        threshold = search_test_threshold(whole_probs, data_ratio, to_dir=to_dir, sub_data_type='whole', update=update)
        record_info['threshold'] = threshold

    result_df = pd.DataFrame({ID: whole_ids})
    result_df = generate_submit(result_df, whole_probs, threshold=threshold)

    if data_type == 'test':
        submit_df = pd.read_csv(opj(DATA_DIR, 'raw', 'sample_submission.csv'))
        result_df = pd.merge(submit_df[[ID]], result_df, on=ID, how='left')
        assert np.all(result_df[PREDICTED].notnull())

        # replace leak result
        result_df = modify_submit(result_df)

    result_df.to_csv(opj(to_dir, 'results_%s.csv.gz' % data_type), index=False, compression='gzip')

    if data_type == 'val':
        acc, kaggle_score, focal_loss = get_kaggle_score(result_df, train_df, whole_probs)
    result_df.to_csv(opj(to_dir, 'results_%s_%.5f.csv.gz' % (data_type, kaggle_score)), index=False, compression='gzip')

    pred_labels = generate_distribution(result_df)
    record_info['%s_num' % data_type] = pred_labels

    info_df = pd.DataFrame(record_info)
    return threshold, kaggle_score, info_df

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    main()
    print('\nsuccess!')
