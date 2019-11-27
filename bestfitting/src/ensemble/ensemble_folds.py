import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
from tqdm import tqdm

from config.config import *
from utils.common_util import *
from config.en_config import *
from utils.leak_util import *
from utils.ensemble_util import *

import argparse
parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--en_cfgs', default='external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds',
                    type=str, help='en configs name')
parser.add_argument('--do_valid', default=1, type=int)
parser.add_argument('--do_test', default=1, type=int)
parser.add_argument('--update', default=0, type=int)

def main():
    args = parser.parse_args()

    en_cfgs = args.en_cfgs
    en_cfgs = eval(en_cfgs)

    do_valid = args.do_valid == 1
    do_test = args.do_test == 1
    update = args.update == 1

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

    to_dir = opj(RESULT_DIR, 'submissions', 'ensemble', args.en_cfgs)
    os.makedirs(to_dir, exist_ok=True)

    threshold = 0.5
    kaggle_score = 0
    info_dfs = [pd.DataFrame(data=LABEL_NAME_LIST, columns=['label_name'])]
    if do_valid:
        data_type = 'val'
        threshold, kaggle_score, val_info_df = ensemble_folds(en_cfgs, train_df, to_dir, test_ratio, data_type=data_type,
                                                              threshold=threshold, kaggle_score=kaggle_score, update=update)
        info_dfs.append(val_info_df)
    if do_test:
        data_type = 'test'
        threshold, kaggle_score, test_info_df = ensemble_folds(en_cfgs, train_df, to_dir, test_ratio, data_type=data_type,
                                                               threshold=threshold, kaggle_score=kaggle_score, update=update)
        info_dfs.append(test_info_df)
    info_df = pd.concat(info_dfs, axis=1)
    for col in info_df.columns:
        if 'num' in col:
            info_df[col] = info_df[col].fillna(0).astype(int)

    print(info_df)
    info_df.to_csv(opj(to_dir, 'record_info.csv'), index=False)

def load_data(en_cfg, data_type='val'):
    model_name = en_cfg['model_name']
    fold = en_cfg['fold']
    epoch_name = en_cfg['epoch_name']
    augment = en_cfg.get('augment', 'whole_maximum')

    sub_dir = opj(RESULT_DIR, 'submissions', model_name, 'fold%d' % fold, 'epoch_%s' % epoch_name, augment)
    df = pd.read_csv(opj(sub_dir, 'results_%s.csv.gz' % data_type))
    probs = np.load(opj(sub_dir, 'prob_%s.npy' % data_type))
    return df, probs

def merge_folds(en_cfgs, data_type='val'):
    if data_type == 'val':
        df_list = []
        prob_list = []
        for en_cfg in en_cfgs:
            df, probs = load_data(en_cfg, data_type=data_type)
            df_list.append(df)
            prob_list.append(probs)
        result_df = pd.concat(df_list, axis=0, ignore_index=True)
        pred_probs = np.vstack(prob_list)
    elif data_type == 'test':
        base_df = None
        base_probs = None
        for en_cfg in en_cfgs:
            df, probs = load_data(en_cfg, data_type=data_type)
            if base_df is None:
                base_df = df
                base_probs = probs
            else:
                assert np.all(base_df[ID].values == df[ID].values)
                base_probs += probs
        base_probs = base_probs / len(en_cfgs)

        result_df = base_df
        pred_probs = base_probs
    else:
        raise ValueError('data_type must be [val, test], not %s', str(data_type))
    return result_df, pred_probs

def ensemble_folds(en_cfgs, train_df, to_dir, data_ratio, data_type='val', threshold=0.5, kaggle_score=0, update=False):
    result_df, pred_probs = merge_folds(en_cfgs, data_type=data_type)

    # average duplicate sample
    if data_type == 'test':
        test_match_df = pd.read_csv(opj(DATA_DIR, 'meta', 'test_match_test.csv.gz'))
        for i in tqdm(range(len(test_match_df))):
            data = test_match_df.iloc[i]
            img_id1 = data['Test1']
            img_id2 = data['Test2']
            ix = (result_df[ID]==img_id1)|(result_df[ID]==img_id2)
            sel_ix = np.where(ix)[0]
            sel_probs = pred_probs[sel_ix]
            mean_probs = np.mean(sel_probs, axis=0)
            pred_probs[sel_ix] = mean_probs

    prob_fname = opj(to_dir, 'prob_%s.npy' % data_type)
    np.save(prob_fname, pred_probs)

    if data_type == 'val':
        train_df = pd.merge(result_df[[ID]], train_df, on=ID, how='left')

    record_info = {}

    # search threshold
    if data_type == 'test':
        threshold = search_test_threshold(pred_probs, data_ratio, to_dir=to_dir, update=update)
        record_info['threshold'] = threshold

    result_df = generate_submit(result_df, pred_probs, threshold=threshold)
    if data_type == 'test':
        submit_df = pd.read_csv(opj(DATA_DIR, 'raw', 'sample_submission.csv'))
        result_df = pd.merge(submit_df[[ID]], result_df, on=ID, how='left')
        assert np.all(result_df[PREDICTED].notnull())

        # replace leak result
        result_df = modify_submit(result_df)

    result_df.to_csv(opj(to_dir, 'results_%s.csv.gz' % data_type), index=False, compression='gzip')

    if data_type == 'val':
        acc, kaggle_score, focal_loss = get_kaggle_score(result_df, train_df, pred_probs)
    result_df.to_csv(opj(to_dir, 'results_%s_%.5f.csv.gz' % (data_type, kaggle_score)), index=False, compression='gzip')

    pred_labels = generate_distribution(result_df)
    record_info['%s_num' % data_type] = pred_labels

    info_df = pd.DataFrame(record_info)
    return threshold, kaggle_score, info_df

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    main()
    print('\nsuccess!')
