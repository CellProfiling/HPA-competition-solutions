import math
import os
from os.path import join as pjoin

import json
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import GPUtil
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import sklearn.metrics

from .config import print_config, class_labels
from .utils import (
    anno_to_binary, cut_score, debug, display_imgs, info, gen_cwd_slash, labels_to_str, load_config, load_img,
    np_macro_f1, str_to_labels, class_id_to_label, class_ids_to_label, combine_windows, chunk, compute_i_coords,
    format_macro_f1_details, vec_to_str
)
# from .utils_heavy import predict, model_from_config
from .ignite_trainer import predict as predict

# def predict_and_save_scores(
#     config,
#     path_to_anno=None,
#     path_to_imgs=None,
#     save_scores_to=None,
#     to_csv=None,
# ):
#     model = model_from_config(config, which='latest')
#     valid_anno = pd.read_csv(path_to_anno, index_col=0)

#     predict(config)

#     return valid_anno_predicted


def remove_scores_predicted(config):
    cwd_slash = gen_cwd_slash(config)
    pd.read_csv(cwd_slash('validation_predictions.csv'), index_col=0) \
      .drop('Scores Predicted', 1) \
      .to_csv(cwd_slash('validation_predictions.csv'))


def evaluate_validation_prediction(config):
    info('evaluate_validation_prediction()')
    cwd_slash = gen_cwd_slash(config)
    anno = pd.read_csv(config['path_to_valid_anno_cache'], index_col=0, dtype=object)
    prediction_df = pd.read_csv(cwd_slash('valid_predicted.csv'), index_col=0, dtype=object)
    anno = anno.join(prediction_df, how='left')
    # DEBUG BEGIN
    anno.loc[:, ['Target', 'Predicted', 'folder', 'extension']].to_csv(cwd_slash('valid_anno_predicted.csv'))
    # DEBUG END
    y_true, y_pred = anno_to_binary(anno, config)
    macro_f1_score, f1_details = np_macro_f1(y_true, y_pred, config, return_details=True)
    print(format_macro_f1_details(f1_details, config))
    print(f'macro_f1_score = {macro_f1_score}')


def final_corrections(config):
    info('final_corrections()')
    cwd_slash = gen_cwd_slash(config)

    anno = pd.read_csv(cwd_slash('test_predicted.csv'), index_col=0)

    # correct best submission [TODO: REMOVE: not for private leaderboard] --------------

    # best_anno = pd.read_csv(cwd_slash('submission_587.csv'), index_col=0)
    # rare_classes = [15, 27, 10, 8, 9, 17, 20, 24, 26]
    # comparison_anno = anno.copy()
    # comparison_anno['best'] = best_anno['Predicted']
    # plot_imgs(
    #     config,
    #     comparison_anno.query('best != Predicted').sample(28),
    #     save_as='./tmp/best_submission_corrections.png',
    #     folder='data/test_minimaps',
    #     extension='jpg',
    # )
    # new_rows = []
    # for id_, row in comparison_anno.iterrows():
    #     current_labels = str_to_labels(row['Predicted'])
    #     best_labels = str_to_labels(row['best'])

    #     for c in rare_classes:
    #         if c in current_labels and c not in best_labels:
    #             debug(f"removing {c} from {id_}")
    #             current_labels.remove(c)

    #         if c not in current_labels and c in best_labels:
    #             debug(f"adding {c} to {id_}")
    #             current_labels.append(c)

    #     new_row = {
    #         'Id': id_,
    #         'Predicted': labels_to_str(current_labels),
    #     }
    #     new_rows.append(new_row)

    # anno = pd.DataFrame.from_records(new_rows).set_index('Id')
    # debug(f"anno ({len(anno)}) =\n{anno.head(10)}")

    # correct leaked --------------

    # pairs_anno = pd.read_csv('data/identical_pairs.csv')
    # hpa_anno = pd.read_csv('data/hpa_public_imgs.csv', index_col=0)
    # correction_anno = pairs_anno.join(hpa_anno, how='left', on=['hpa_id'])\
    #                             .join(anno, how='left', on=['test_id'])
    # correction_anno['Target'] = [labels_to_str(str_to_labels(x)) for x in correction_anno['Target']]

    # debug(f"correction_anno['test_id'] = {correction_anno['test_id']}")
    # debug(f"len = {len(anno.loc[correction_anno['test_id'], 'Predicted'].values)}")

    # correction_anno['Predicted'] = anno.loc[correction_anno['test_id'], 'Predicted'].values
    # actual_corrections = correction_anno.query('Predicted != Target').set_index('test_id')
    # # DEBUG BEGIN
    # # plot_imgs(config, actual_corrections, folder='data/test_minimaps', extension='jpg')
    # # DEBUG END
    # debug(f"making {len(correction_anno)} corrections, {len(actual_corrections)} are actually different")
    # debug(f"actual_corrections =\n{actual_corrections}")
    # anno.loc[correction_anno['test_id'], 'Predicted'] = correction_anno['Target'].values

    # correct leaked 2 --------------

    pairs_anno = pd.read_csv('data/identical_pairs_new_fixed.csv')
    for i_begin, i_end in chunk(len(pairs_anno), 24):
        plot_imgs(
            config,
            pairs_anno.iloc[i_begin:i_end].drop('test_id', axis=1).set_index('hpa_id'),
            save_as=f'./tmp/diff_{i_begin}_hpa.jpg',
            folder='data/hpa_public_imgs',
            extension='jpg',
            background_color=None,
            channel=None,
            dpi=100,
        )
        plot_imgs(
            config,
            pairs_anno.iloc[i_begin:i_end].drop('hpa_id', axis=1).set_index('test_id'),
            save_as=f'./tmp/diff_{i_begin}_test.jpg',
            folder='data/test_full_size',
            extension='tif',
            background_color=None,
            channel=['red', 'green', 'blue'],
            dpi=100,
        )
    hpa_anno = pd.read_csv('data/hpa_public_imgs.csv', index_col=0)
    correction_anno = pairs_anno.join(hpa_anno, how='left', on=['hpa_id'])\
                                .join(anno, how='left', on=['test_id'])
    correction_anno['Target'] = [labels_to_str(str_to_labels(x)) for x in correction_anno['Target']]

    debug(f"correction_anno['test_id'] = {correction_anno['test_id']}")
    debug(f"len = {len(anno.loc[correction_anno['test_id'], 'Predicted'].values)}")

    correction_anno['Predicted'] = anno.loc[correction_anno['test_id'], 'Predicted'].values
    actual_corrections = correction_anno.query('Predicted != Target').set_index('test_id')
    # DEBUG BEGIN
    # plot_imgs(config, actual_corrections, folder='data/test_minimaps', extension='jpg')
    # DEBUG END
    debug(f"making {len(correction_anno)} corrections, {len(actual_corrections)} are actually different")
    debug(f"actual_corrections =\n{actual_corrections}")
    anno.loc[correction_anno['test_id'], 'Predicted'] = correction_anno['Target'].values

    # DEBUG BEGIN
    # plot_imgs(
    #     config,
    #     anno.loc[[27 in str_to_labels(p) for p in anno['Predicted']]],
    #     folder='data/test_minimaps',
    #     extension='jpg'
    # )
    # DEBUG END
    anno.to_csv(cwd_slash('test_predicted_corrected.csv'))


# def list_confusion(config):
#     fn_counts_list = {}
#     class_labels = [f'{k}-{classes[k]}' for k in range(n_classes)]
#     for which_class in tqdm(range(n_classes)):
#         cwd_slash = gen_cwd_slash(config)
#         anno = pd.read_csv(cwd_slash('validation_predictions.csv'), index_col=0)
#         y_true, y_pred = anno_to_binary(anno)
#         fn = y_true * (1 - y_pred)
#         fp = (1 - y_true) * y_pred
#         i_fn_predictions = np.nonzero(fn[:, which_class])[0]
#         fn_counts = fp[i_fn_predictions, :].sum(axis=0) / len(i_fn_predictions)
#         fn_counts_list[class_labels[which_class]] = fn_counts
#         # out = pd.Series(fn_counts, index=pd.Index(range(n_classes), name='class'))\
#         #         .sort_values(ascending=False)\
#         #         .head(3)

#     pd.DataFrame(fn_counts_list, index=class_labels).to_csv('./tmp/confusion.csv')


def plot_imgs(
    config,
    anno,
    save_as='./tmp/imgs.jpg',
    folder=None,
    extension=None,
    background_color=None,
    channel=None,
    dpi=100,
):
    img_list = []
    for id_, row in anno.iterrows():
        img = load_img(
            id_,
            config,
            resize=False,
            folder=row.get('folder') or folder,
            channel=channel,
            extension=row.get('extension') or extension,
        )
        # if type(channel) is str:
        #     channel = {
        #         'red': 0,
        #         'green': 1,
        #         'blue': 2,
        #         'yellow': 3,
        #     }.get(channel)

        # if channel is not None:
        #     img = img[:, :, channel]

        debug(f'  - Loaded image {id_} with size {img.shape}')
        img_label = '\n'.join([f'{id_}'] + [f'{k} = {v}' for k, v in row.items()])
        img_list.append((img, img_label))

    display_imgs(
        img_list,
        save_as=save_as,
        background_color=background_color,
        dpi=dpi,
    )


def plot_tfpn_examples(config, which_class, max_n_imgs=28, output_folder='./tmp'):
    cwd_slash = gen_cwd_slash(config)
    anno = pd.read_csv(cwd_slash('validation_predictions.csv'), index_col=0)
    y_true, y_pred = anno_to_binary(anno)
    y_true = y_true[:, which_class]
    y_pred = y_pred[:, which_class]

    def plot_imgs(selector, filename, background_color):
        debug(f'selector = {selector}')
        if type(config['score_threshold']) is list:
            score_threshold = config['score_threshold'][which_class]
        else:
            score_threshold = config['score_threshold']
        tp_idxs = np.nonzero(selector > score_threshold)[0]
        if len(tp_idxs) > max_n_imgs:
            sample_idxs = np.sort(np.random.choice(range(len(tp_idxs)), max_n_imgs, replace=False))
            tp_idxs = tp_idxs[sample_idxs]

        img_list = []
        for idx in tp_idxs:
            row = anno.iloc[idx]
            img_id = row.name
            labels_true = class_ids_to_label(str_to_labels(row['Target']), config)
            labels_pred = class_ids_to_label(str_to_labels(row['Predicted']), config)

            img_label = '\n'.join([
                f'{img_id}',
                f'T: {labels_true}',
                f'P: {labels_pred}',
            ])

            # img = load_img(img_id, self.config, resize=False, folder='./data/train_full_size', extension='tif')
            img = load_img(
                img_id,
                config,
                resize=False,
                folder=config['path_to_valid'],
                channel=None,
                extension=config['img_extension'],
            )
            debug(f'  - Loaded image {img_id} with size {img.shape}')
            img_list.append((img, img_label))

        display_imgs(
            img_list,
            save_as=filename,
            background_color=background_color,
        )

    def out_slash(fn):
        return pjoin(output_folder, fn)

    plot_imgs(y_true * y_pred, out_slash(f'class_{which_class}_true_positives.png'), 'white')
    plot_imgs((1 - y_true) * y_pred, out_slash(f'class_{which_class}_false_positives.png'), 'yellow')
    plot_imgs(y_true * (1 - y_pred), out_slash(f'class_{which_class}_false_negatives.png'), 'blue')
    # plot_imgs((1 - y_true) * (1 - y_pred), out_slash(f'class_{which_class}_true_negatives.png'), 'black')


def add_extra_data_into_train_anno(config):
    cwd_slash = gen_cwd_slash(config)
    train_anno = pd.read_csv(cwd_slash('train_windowed_anno.csv'), index_col=0)
    valid_anno = pd.read_csv(cwd_slash('valid_windowed_anno.csv'), index_col=0)
    train_with_hpa_anno = pd.read_csv('data/train_with_hpa.csv', index_col=0)
    train_windowed_anno = pd.read_csv('data/train_windowed.csv', index_col=0)

    hpa_ids = set(train_with_hpa_anno.index)
    existing_ids = set(valid_anno['source_img_id']).union(train_anno['source_img_id'])
    new_ids = hpa_ids.difference(existing_ids)
    extra_train_anno = train_with_hpa_anno.loc[new_ids]
    debug(f'extra_train_anno ({len(extra_train_anno)}) =\n{extra_train_anno.head(10)}')
    extra_train_windowed_anno = train_windowed_anno.join(extra_train_anno, how='right', on=['source_img_id'])
    debug(f'extra_train_windowed_anno ({len(extra_train_windowed_anno)}) =\n{extra_train_windowed_anno.head(10)}')
    pd.concat([train_anno, extra_train_windowed_anno]).to_csv(cwd_slash('train_windowed_anno.csv'))


# def calibrate_one_task(task):
#     i_class = task['i_class']
#     mat_pred_windowed = task['mat_pred_windowed']
#     mat_true = task['mat_true']
#     alpha = task['alpha']
#     i_windowss = task['i_windowss']
#     beta_values = task['beta_values']
#     config = task['config']

#     details_list = []

#     for beta in beta_values:
#         vec_true = mat_true[:, i_class]
#         vec_pred_windowed = mat_pred_windowed[:, i_class]

#         list_pred = []
#         for i_source, i_windows in enumerate(i_windowss):
#             combined_prediction = vec_pred_windowed[i_windows].mean() + vec_pred_windowed[i_windows].mean()
#             list_pred.append(combined_prediction)
#         vec_pred = np.array(list_pred)

#         f1 = np_macro_f1(vec_true, vec_pred, config)
#         details_list.append({
#             'i_class': i_class,
#             'alpha': alpha,
#             'beta': beta,
#             'f1': f1,
#         })

#         # debug(f'i_class = {i_class}, alpha = {alpha}, beta = {beta}, f1 = {f1}, best_f1 = {best_f1}')

#     details_df = pd.DataFrame.from_records(details_list)

#     return {
#         'task': task,
#         'details_df': details_df,
#     }

# def calibrate_windowed_score(
#     config,
#     n_threads=70,
#     n_cols=7,
#     save_graph_to='./tmp/calibrate_score_threshold.png',
#     epsilon=1e-7,
# ):
#     info('calibrate_windowed_score()')
#     cwd_slash = gen_cwd_slash(config)

#     alpha_values = range(10)
#     beta_values = np.linspace(0, 1, 21)

#     mat_pred_windowed = np.load(cwd_slash('valid_windowed_scores.npy'))
#     valid_anno = pd.read_csv(config['path_to_valid_anno_cache'])
#     mat_true = np.zeros((valid_anno.shape[0], 28))
#     for i, target_str in enumerate(valid_anno['Target']):
#         targets = str_to_labels(target_str)
#         mat_true[np.ix_([i], targets)] = 1

#     valid_windowed_anno = pd.read_csv(cwd_slash('valid_windowed_anno.csv'))
#     valid_windowed_anno['row_number'] = valid_windowed_anno.index
#     grouped = valid_windowed_anno.groupby('source_img_id')
#     source_id_to_window_row_nums = {id_: group['row_number'].values.tolist() for id_, group in grouped}
#     i_windowss = [source_id_to_window_row_nums[id_] for id_ in valid_anno['Id']]

#     task_list = [
#         {
#             'i_class': i_class,
#             'alpha': alpha,
#             'mat_pred_windowed': mat_pred_windowed,
#             'mat_true': mat_true,
#             'i_windowss': i_windowss,
#             'beta_values': beta_values,
#             'config': config,
#         } for i_class in range(config['_n_classes']) for alpha in alpha_values
#     ]

#     details_dfs = []
#     with Pool(n_threads) as p:
#         result_iter = p.imap_unordered(calibrate_one_task, task_list)
#         for i_result, result in enumerate(result_iter):
#             info(
#                 f"({i_result}/{len(task_list)}) "
#                 f"i_class = {result['task']['i_class']}, "
#                 f"alpha = {result['task']['alpha']} is done"
#             )
#             details_dfs.append(result['details_df'])
#     details_df = pd.concat(details_dfs)

#     if save_graph_to is not None:
#         n_rows = math.ceil(config['_n_classes'] / n_cols)
#         plt.figure(figsize=(n_cols * 10, n_rows * 10))
#         for i_class, group_df in details_df.groupby('i_class'):
#             mat = group_df.pivot(index='beta', columns='alpha', values='f1')
#             plt.subplot(n_rows, n_cols, i_class + 1, sharex=plt.gca(), sharey=plt.gca())
#             plt.imshow(mat, aspect='auto')
#             plt.xticks(range(len(alpha_values)), alpha_values)
#             plt.yticks(range(len(beta_values)), beta_values)
#             plt.text(0, 1, f'{i_class}', transform=plt.gca().transAxes)
#         plt.savefig(save_graph_to, dpi=100)
#         debug(f'Saved graph to {save_graph_to}')

#     print(details_df)
#     details_df.to_csv(cwd_slash('calibrate_windowed_score_details.csv'), index=False)
#     debug(f"saved to {cwd_slash('calibrate_windowed_score_details.csv')}")

#     best_df = pd.concat([group.sort_values('f1').tail(1) for i_class, group in details_df.groupby('i_class')])
#     best_df['manually_modified'] = False
#     best_df.to_csv(cwd_slash('calibrate_windowed_score.csv'), index=False)
#     debug(f"saved to {cwd_slash('calibrate_windowed_score.csv')}")

# def calibrate_score_threshold(config, n_cols=7, save_graph_to='./tmp/calibrate_score_threshold.png', epsilon=1e-7):
#     info('calibrate_score_threshold()')
#     cwd_slash = gen_cwd_slash(config)

#     n_rows = math.ceil(config['_n_classes'] / n_cols)

#     mat_pred = np.load(cwd_slash('valid_scores.npy'))

#     anno = pd.read_csv(cwd_slash('valid_windowed_anno.csv'))
#     mat_true = np.zeros_like(mat_pred)
#     for i, target_str in enumerate(anno['Target']):
#         targets = str_to_labels(target_str)
#         mat_true[np.ix_([i], targets)] = 1

#     if save_graph_to is not None:
#         plt.figure(figsize=(n_cols * 10, n_rows * 10))

#     best_ths = []
#     for class_id in tqdm(config['classes']):
#         thresholds = np.round(np.linspace(0, 1, 1001), 3)
#         f1_scores = np.zeros_like(thresholds)
#         ps = []
#         rs = []
#         for i_th, th in enumerate(thresholds):
#             y_pred = mat_pred[:, i_class]
#             y_pred = np.where(y_pred < th, np.zeros_like(y_pred), np.ones_like(y_pred))
#             y_true = mat_true[:, i_class]

#             tp = np.sum(y_true * y_pred, axis=0)
#             # tn = np.sum((1 - y_true) * (1 - y_pred), axis=0)
#             fp = np.sum((1 - y_true) * y_pred, axis=0)
#             fn = np.sum(y_true * (1 - y_pred), axis=0)

#             p = tp / (tp + fp + epsilon)
#             r = tp / (tp + fn + epsilon)

#             ps.append(p)
#             rs.append(r)

#             out = 2 * p * r / (p + r + epsilon)
#             # replace all NaN's with 0's
#             out = np.where(np.isnan(out), np.zeros_like(out), out)
#             f1_scores[i_th] = out

#         if save_graph_to is not None:
#             plt.subplot(n_rows, n_cols, i_class + 1, sharex=plt.gca(), sharey=plt.gca())
#             plt.plot(thresholds, f1_scores)
#             plt.plot(thresholds, ps)
#             plt.plot(thresholds, rs)
#             plt.text(0, 1, f'{i_class}', transform=plt.gca().transAxes)
#         # debug(f'thresholds = {thresholds}')
#         # debug(f'f1_scores = {f1_scores}')
#         best_th = thresholds[np.argmax(f1_scores)]
#         best_ths.append(best_th)

#     if save_graph_to is not None:
#         plt.savefig(save_graph_to, dpi=100)
#         debug(f'Saved graph to {save_graph_to}')

#     debug(f'best_ths = {best_ths}')

#     with open(cwd_slash('calibrated_score_threshold.json'), 'w') as f:
#         json.dump(best_ths, f)


def predict_for_valid(config):
    cwd_slash = gen_cwd_slash(config)

    valid_windowed_anno = pd.read_csv(config['path_to_valid_windowed_anno_cache'], index_col=0)

    predict(
        config,
        valid_windowed_anno,
        cwd_slash('model.pth'),
        save_numpy_to='valid_windowed_predicted.npy',
        save_csv_to='valid_windowed_anno_predicted.csv',
        target_col='corrected_target',
    )

    # predict(
    #     anno=cwd_slash('valid_windowed_anno.csv'),
    #     config=config,
    #     extension=config['img_extension'],
    #     folder=config['path_to_valid'],
    #     to_npy=cwd_slash('valid_windowed_scores.npy'),
    # )


# def cut_score_for_valid(config):
#     info('cut_score_for_valid()')
#     cwd_slash = gen_cwd_slash(config)

#     path_to_score = cwd_slash('calibrate_windowed_score.csv')
#     if os.path.exists(path_to_score):
#         tb = pd.read_csv(path_to_score)
#         debug(f"read from {path_to_score}")
#         score_threshold = tb.sort_values('i_class')['beta'].values
#         debug(f'score_threshold = {score_threshold}')
#         min_n_windows = tb.sort_values('i_class')['alpha'].values
#         debug(f'min_n_windows = {min_n_windows}')
#     else:
#         debug(f'WARNING: using default score_threshold and min_n_windows')
#         score_threshold = config['score_threshold']
#         min_n_windows = 3

#     # if os.path.exists(cwd_slash('calibrated_score_threshold.json')):
#     #     with open(cwd_slash('calibrated_score_threshold.json'), 'r') as f:
#     #         score_threshold = json.load(f)
#     # else:
#     #     score_threshold = config['score_threshold']

#     debug('cut_score()')
#     cut_score(
#         anno=cwd_slash('valid_windowed_anno.csv'),
#         scores_mat=cwd_slash('valid_windowed_scores.npy'),
#         config=config,
#         prediction_col='Predicted',
#         score_threshold=score_threshold,
#         to_csv=cwd_slash('valid_windowed_predicted.csv'),
#     )

#     debug('combine_windows()')
#     combine_windows(
#         cwd_slash('valid_windowed_predicted.csv'),
#         min_n_windows,
#         config,
#         save_combined_anno_to=cwd_slash('valid_predicted.csv'),
#         group_col='source_img_id',
#     )


def predict_for_test(config):
    info('predict_for_test()')
    cwd_slash = gen_cwd_slash(config)

    test_windowed_anno = pd.read_csv(config['path_to_test_anno'], index_col=0)
    test_windowed_anno = compute_i_coords(test_windowed_anno, config)
    test_windowed_anno['group'] = 'test_full_size'

    predict(
        config,
        test_windowed_anno,
        cwd_slash('model.pth'),
        save_numpy_to='test_windowed_predicted.npy',
        save_csv_to='test_windowed_anno_predicted.csv',
    )

    # anno = pd.read_csv('./data/test_windowed.csv', index_col=0)
    # if config['submission_subsampling'] is not None:

    #     anno = anno.sample(config['submission_subsampling'])

    # predict(
    #     anno=anno,
    #     config=config,
    #     extension=config['img_extension'],
    #     folder=config['path_to_test'],
    #     to_npy=cwd_slash('test_windowed_scores.npy'),
    # )


def create_csv_for_debugger(config):
    info('create_csv_for_debugger()')
    cwd_slash = gen_cwd_slash(config)

    anno = pd.read_csv(cwd_slash('valid_windowed_anno.csv'), index_col=0)

    pred_mat = np.load(cwd_slash('valid_windowed_scores.npy'))
    pred_anno = pd.DataFrame(pred_mat, columns=[f'score_of_{x}' for x in config['class_ids']], index=anno.index)

    anno.join(pred_anno, how='left').to_csv(cwd_slash('valid_windowed_scores.csv'))


def take_top_n_for_test(config):
    info('take_top_n_for_test()')
    cwd_slash = gen_cwd_slash(config)
    class_distrib = pd.read_csv('tmp/class_distribution.csv', index_col=0)
    # test_scores = pd.read_csv(cwd_slash('test_aggregated_prediction.csv'), index_col=0)
    test_scores = pd.read_csv(cwd_slash('stacking_v3_test.csv'), index_col=0)

    # correct class 17 for LB613
    # test_scores_LB613 = pd.read_csv(
    #     './working/__613__190104-001629__P1T500_/test_aggregated_prediction.csv', index_col=0
    # )
    # test_scores_LB613['17'] = test_scores['17']
    # test_scores = test_scores_LB613

    # test_scores = pd.read_csv('tmp/yuanhao.csv', index_col=0)
    submission_df = pd.read_csv('data/sample_submission.csv', index_col=0)
    test_scores = test_scores.loc[submission_df.index]

    def get_order(col):
        fixed_n_samples = class_distrib.loc[int(col.name), 'LB613']
        if not np.isnan(fixed_n_samples):
            n_samples = fixed_n_samples
        else:
            n_samples = class_distrib.loc[int(col.name), 'expected_n_samples_in_test'] * 1.2
        return np.where(np.argsort(np.argsort(-col)) >= n_samples, 0, 1)

    submission_df['Predicted'] = test_scores.apply(get_order).apply(vec_to_str, axis=1)
    submission_df.to_csv(cwd_slash('submission_top_n_stacking_v3.csv'))
    # submission_df.to_csv('tmp/yuanhao_submission.csv')


def cut_score_for_valid(config):
    info('cut_score_for_valid()')
    cwd_slash = gen_cwd_slash(config)

    threshold_df = pd.read_csv(cwd_slash('calibrated_threshold_17_corrected.csv'), index_col=0)
    thresholds = threshold_df['best_threshold']
    valid_scores = pd.read_csv(cwd_slash('valid_aggregated_prediction_17_corrected.csv'), index_col=0)
    submission_df = pd.read_csv(cwd_slash('valid_anno.csv'))
    valid_scores = valid_scores.loc[submission_df['Id']]

    pick_mat = valid_scores.values > [thresholds]
    preds = [vec_to_str(row) for row in pick_mat]

    submission_df['Predicted'] = preds
    submission_df.to_csv(cwd_slash('valid_anno_predicted.csv'), index=False)


def cut_score_for_test(config):
    info('cut_score_for_test()')
    cwd_slash = gen_cwd_slash(config)

    threshold_df = pd.read_csv(cwd_slash('calibrated_threshold.csv'), index_col=0)
    # thresholds = threshold_df['best_threshold'] * 0.4
    test_scores = pd.read_csv(cwd_slash('test_aggregated_prediction.csv'), index_col=0)
    submission_df = pd.read_csv('data/sample_submission.csv')
    test_scores = test_scores.loc[submission_df['Id']]

    pick_mat = test_scores.values > [thresholds]

    def get_order(col, n_samples):
        return np.where(np.argsort(np.argsort(-col)) >= n_samples, 0, 1)

    for class_id in test_scores:
        i_class = int(class_id)
        manual_top_n = threshold_df.loc[i_class, 'manual_top_n']
        if not np.isnan(manual_top_n):
            debug(f"manually set {class_id} to pick the top {manual_top_n}")
            pick_vec = get_order(test_scores[class_id], manual_top_n)
            pick_mat[:, i_class] = pick_vec

    preds = [vec_to_str(row) for row in pick_mat]

    submission_df['Predicted'] = preds
    submission_df.to_csv(cwd_slash('submission.csv'), index=False)


def compare_with_best_submssion(config):
    info('compare_with_best_submssion()')
    cwd_slash = gen_cwd_slash(config)

    best_submission = pd.read_csv(cwd_slash('submission_587.csv'), index_col=0)
    current_submission = pd.read_csv(cwd_slash('test_predicted_corrected.csv'), index_col=0)
    current_submission['best'] = best_submission['Predicted']
    debug(f"index all equal = {np.all(current_submission.index.values == best_submission.index.values)}")
    diff = current_submission.query('Predicted != best')
    # DEBUG BEGIN
    plot_imgs(
        config,
        diff.loc[[10 in (str_to_labels(row['Predicted']) + str_to_labels(row['best'])) for i, row in diff.iterrows()]],
        folder='data/test_minimaps',
        extension='jpg',
        channel='green',
    )
    # DEBUG END
    # debug(f"diff =\n{diff}")
    save_path = './tmp/diff.csv'
    diff.to_csv(save_path)
    debug(f"saved to {save_path}")


def show_score_details(config, id_='94c0f350-bada-11e8-b2b9-ac1f6b6435d0'):
    info('show_score_details()')
    cwd_slash = gen_cwd_slash(config)

    windowed_anno = pd.read_csv('./data/test_windowed.csv')
    scores_mat = np.load(cwd_slash('test_windowed_scores.npy'))

    idxs = windowed_anno.loc[windowed_anno['source_img_id'] == id_].index.values

    print(pd.DataFrame(np.round(scores_mat[idxs, :], 3), index=windowed_anno.loc[idxs]['img_id']))


def aggregate_prediction_for_valid(config):
    info('aggregate_prediction_for_valid()')
    cwd_slash = gen_cwd_slash(config)

    anno = pd.read_csv(cwd_slash('valid_windowed_anno_predicted.csv'))
    score_cols = [str(class_id) for class_id in config['class_ids']]
    anno_agg = anno.groupby('source_img_id')[score_cols].agg([np.mean, np.max])
    result_df = pd.DataFrame(index=anno_agg.index)
    for score_col in score_cols:
        result_df[score_col] = (anno_agg[score_col, 'mean'] + anno_agg[score_col, 'amax']) / 2
        # result_df[score_col] = anno_agg[score_col, 'mean']
    print(result_df.head())
    save_path = cwd_slash('valid_aggregated_prediction.csv')
    result_df.to_csv(save_path)
    debug(f"saved to {save_path}")


def aggregate_prediction_for_test(config):
    info('aggregate_prediction_for_test()')
    cwd_slash = gen_cwd_slash(config)

    anno = pd.read_csv(cwd_slash('test_windowed_anno_predicted.csv'))
    score_cols = [str(class_id) for class_id in config['class_ids']]
    anno_agg = anno.groupby('source_img_id')[score_cols].agg([np.mean, np.max])
    result_df = pd.DataFrame(index=anno_agg.index)
    for score_col in score_cols:
        result_df[score_col] = (anno_agg[score_col, 'mean'] + anno_agg[score_col, 'amax']) / 2
        # result_df[score_col] = anno_agg[score_col, 'mean']
    print(result_df.head())
    save_path = cwd_slash('test_aggregated_prediction.csv')
    result_df.to_csv(save_path)
    debug(f"saved to {save_path}")


def calibrate_thresholds(config, epsilon=1e-7):
    info('calibrate_thresholds()')
    cwd_slash = gen_cwd_slash(config)

    anno = pd.read_csv(config['path_to_valid_anno_cache'], index_col=0)

    pred_df = pd.read_csv(cwd_slash('valid_aggregated_prediction.csv'), index_col=0).sort_index()

    truth_rows = []
    for id_, row in anno.iterrows():
        labels = str_to_labels(row['Target'])
        truth_row = np.array([class_id in labels for class_id in config['class_ids']]).astype(int)
        truth_row = pd.Series(truth_row, name=id_)
        truth_rows.append(truth_row)

    truth_df = pd.concat(truth_rows, axis=1).transpose().sort_index()
    truth_df.columns = config['class_ids']

    macro_f1, details = np_macro_f1(pred_df.values, truth_df.values, config, return_details=True)
    print(format_macro_f1_details(details, config))
    debug(f"macro_f1 = {macro_f1}")

    os.makedirs(cwd_slash('threshold_calibration_dfs'), exist_ok=True)
    plt.figure(figsize=(20, 15))
    threshold_rows = {}
    for class_id in config['class_ids']:
        y_pred = pred_df.iloc[:, class_id]
        y = truth_df[class_id]
        compare_df = pd.DataFrame({
            'y_pred': y_pred,
            'y': y,
        })
        compare_df = compare_df.sort_values('y_pred')
        compare_df['tn'] = (1 - compare_df['y']).cumsum()
        compare_df['fn'] = compare_df['y'].cumsum()

        compare_df = compare_df.sort_values('y_pred', ascending=False)
        compare_df['fp'] = np.concatenate([[0], (1 - compare_df['y']).cumsum()[:-1]])
        compare_df['tp'] = np.concatenate([[0], compare_df['y'].cumsum()[:-1]])

        compare_df['precision'] = compare_df['tp'] / (compare_df['tp'] + compare_df['fp'] + epsilon)
        compare_df['recall'] = compare_df['tp'] / (compare_df['tp'] + compare_df['fn'] + epsilon)
        compare_df['f1'] = 2 * compare_df['precision'] * compare_df['recall'] / (
            compare_df['precision'] + compare_df['recall'] + epsilon
        )
        compare_df['f1_smoothed'] = np.convolve(compare_df['f1'], np.ones(1) / 1, mode='same')
        best_row_idx = compare_df['f1_smoothed'].idxmax()
        picked_threshold = compare_df['y_pred'][best_row_idx]
        best_f1 = compare_df['f1'][best_row_idx]
        threshold_rows[class_id] = compare_df.loc[best_row_idx].copy()

        precisions, recalls, _ = sklearn.metrics.precision_recall_curve(y, y_pred)
        threshold_rows[class_id]['rp_auc'] = auc = sklearn.metrics.auc(recalls, precisions)
        plt.plot(
            recalls,
            precisions,
            label=f"{class_id_to_label(class_id, config)} (rp_auc={auc:.4f})",
        )

        compare_df.to_csv(
            cwd_slash('threshold_calibration_dfs', f"{class_id:02d}_t{picked_threshold:.4f}_f1{best_f1:.4f}.csv")
        )

    threshold_df = pd.DataFrame(threshold_rows).transpose()
    threshold_df.index.name = 'class_id'
    threshold_df['best_threshold'] = threshold_df['y_pred']
    threshold_df['manual_top_n'] = np.nan
    threshold_df['manual_top_n'][15] = 5
    threshold_df['manual_top_n'][27] = 9
    threshold_df['manual_top_n'][20] = 59
    threshold_df['manual_top_n'][8] = 12
    threshold_df['manual_top_n'][9] = 20
    threshold_df['manual_top_n'][10] = 16
    threshold_df['manual_top_n'][16] = 204
    threshold_df['manual_top_n'][17] = 300
    threshold_df = threshold_df[[
        'best_threshold',
        'manual_top_n',
        'tn',
        'fn',
        'fp',
        'tp',
        'precision',
        'recall',
        'f1',
        'f1_smoothed',
        'rp_auc',
    ]]
    print(threshold_df)
    print(f"best total f1: {threshold_df['f1'].mean()}")

    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"{config['job_title']} (mean rp_auc={threshold_df['rp_auc'].mean():.4f})")
    save_path = cwd_slash('recall_precision_curve.png')
    plt.savefig(save_path, dpi=200)
    debug(f"saved to {save_path}")

    threshold_df.to_csv(cwd_slash('calibrated_threshold.csv'))

    # for class_id in config['class_ids']:
    #     pd.DataFrame({'y_pred': pred_df[class_id].sort_values(), 'y':


def logit(x, epsilon=1e-15):
    return x
    x = np.clip(x, epsilon, 1 - epsilon)
    return np.log(x / (1 - x))


def plot_score_distribution(config):
    info('plot_score_distribution()')
    cwd_slash = gen_cwd_slash(config)

    thdf = pd.read_csv(cwd_slash('calibrated_threshold.csv'), index_col=0)
    thdf.index = thdf.index.astype(str)

    valid_agg = pd.read_csv(cwd_slash('valid_aggregated_prediction.csv'))
    valid_agg = valid_agg.melt(id_vars='source_img_id', var_name='class_id', value_name='p')
    valid_agg['logit'] = logit(valid_agg['p'])
    valid_grouped = valid_agg.groupby('class_id')

    test_agg = pd.read_csv(cwd_slash('test_aggregated_prediction.csv'))
    test_agg = test_agg.melt(id_vars='source_img_id', var_name='class_id', value_name='p')
    test_agg['logit'] = logit(test_agg['p'])
    test_grouped = test_agg.groupby('class_id')

    mean_std_df = pd.DataFrame(
        {
            'valid_mean': valid_grouped['logit'].mean(),
            'valid_std': valid_grouped['logit'].std(),
            'test_mean': test_grouped['logit'].mean(),
            'test_std': test_grouped['logit'].std(),
        },
    )

    thdf = thdf.join(mean_std_df, how='left')
    thdf['th_logit'] = logit(thdf['best_threshold'])
    thdf['z_score'] = (thdf['th_logit'] - thdf['valid_mean']) / thdf['valid_std']
    thdf['th_adjusted_logit'] = thdf['z_score'] * thdf['test_std'] + thdf['test_mean']
    thdf['th_adjusted'] = np.exp(thdf['th_adjusted_logit']) / (np.exp(thdf['th_adjusted_logit']) + 1)
    print(thdf)

    valid_agg['group'] = 'valid'
    test_agg['group'] = 'test'
    both_agg = pd.concat([valid_agg, test_agg])
    both_agg['class_id'] = [f"{int(xx):02d}-{class_labels[int(xx)]}" for xx in both_agg['class_id']]

    plt.figure(figsize=(30, 16))
    sns.set_style("whitegrid")
    sns.stripplot(x='class_id', y='logit', hue='group', data=both_agg, jitter=0.3, alpha=0.3, size=1, dodge=True)
    # sns.violinplot(x='class_id', y='logit', hue='group', data=both_agg, inner=None, color='.8', cut=0, bw=0.001)

    for id_, row in thdf.iterrows():
        plt.plot([int(id_) - 0.4, int(id_)], [logit(row['best_threshold']), logit(row['best_threshold'])])
        plt.plot([int(id_), int(id_) + 0.4], [logit(row['best_threshold'] * 0.5), logit(row['best_threshold'] * 0.5)])
        plt.plot([int(id_), int(id_) + 0.4], [logit(row['th_adjusted']), logit(row['th_adjusted'])], dashes=[1, 1])

    plt.axhline(0, dashes=[2, 1, 1, 1])

    save_path = cwd_slash('score_distribution.png')

    plt.xticks(rotation=30, verticalalignment='top', horizontalalignment='right')
    plt.savefig(save_path, dpi=300)
    debug(f"saved to {save_path}")


def aggregate_folds_test_preds(config):
    info('aggregate_folds_test_preds()')
    cwd_slash = gen_cwd_slash(config)

    debug(f"combining {config['n_folds']} folds ...")
    dfs = []
    for i_fold in range(config['n_folds']):
        path_to_scores = cwd_slash(f"fold_{i_fold}", 'test_windowed_anno_predicted.csv')
        anno_predicted = pd.read_csv(path_to_scores)
        df = anno_predicted.melt(
            id_vars=['img_id'],
            value_vars=[str(x) for x in config['class_ids']],
            var_name='class_id',
            value_name='score',
        )
        df['class_id'] = df['class_id'].astype(int)
        df['i_fold'] = i_fold
        dfs.append(df)

    df_all_folds = pd.concat(dfs)

    debug(f"calculating mean across folds ...")
    df_sums = df_all_folds.groupby(['img_id', 'class_id'])['score'].mean()

    combined_scores = df_sums.unstack('class_id')
    debug(f"left join to anno ...")

    combined_anno = anno_predicted.drop(map(str, config['class_ids']), axis=1)
    combined_anno = combined_anno.join(combined_scores, how='left', on='img_id')
    save_path = cwd_slash('test_windowed_anno_predicted.csv')
    combined_anno.to_csv(save_path, index=False)
    print(combined_anno.head())
    debug(f"saved to {save_path}")


def concat_folds_annos(config):
    info('concat_folds_annos()')
    cwd_slash = gen_cwd_slash(config)

    valid_aggregated_predictions = []
    valid_annos = []
    for i_fold in range(config['n_folds']):
        path_to_valid_anno = cwd_slash(f"fold_{i_fold}", 'valid_anno.csv')
        valid_annos.append(pd.read_csv(path_to_valid_anno))

        path_to_valid_aggregated_prediction = cwd_slash(f"fold_{i_fold}", 'valid_aggregated_prediction.csv')
        valid_aggregated_predictions.append(pd.read_csv(path_to_valid_aggregated_prediction))

    valid_anno = pd.concat(valid_annos).sort_values('Id')
    save_path = cwd_slash('valid_anno.csv')
    valid_anno.to_csv(save_path, index=False)
    debug(f"saved to {save_path}")

    valid_aggregated_prediction = pd.concat(valid_aggregated_predictions).sort_values('source_img_id')
    save_path = cwd_slash('valid_aggregated_prediction.csv')
    valid_aggregated_prediction.to_csv(save_path, index=False)
    debug(f"saved to {save_path}")

    # df_all_folds = df_all_folds.sort_values(['img_id', 'class_id', 'i_fold'])
    # print(df_all_folds)


def after_training(config):
    # print_config(config)

    # predict_for_valid(config)
    # calibrate_windowed_score(config)
    # cut_score_for_valid(config)

    # evaluate_validation_prediction(config)

    # predict_for_test(config)
    # cut_score_for_test(config)

    # compare_with_best_submssion(config)
    pass


def after_training_for_folds(config):
    # for i_fold in range(config['n_folds']):
    #     config_for_one_fold = load_config(pjoin(config['_cwd'], f'fold_{i_fold}'))
    #     steps_for_one_fold(config_for_one_fold)

    # aggregate_folds_test_preds(config)
    # concat_folds_annos(config)
    # aggregate_prediction_for_test(config)
    # calibrate_thresholds(config)
    # cut_score_for_test(config)
    # cut_score_for_valid(config)
    take_top_n_for_test(config)

    # plot_score_distribution(config)


def steps_for_one_fold(config):
    # add_extra_data_into_train_anno(config)
    # remove_scores_predicted(config)

    predict_for_valid(config)
    aggregate_prediction_for_valid(config)
    calibrate_thresholds(config)

    # calibrate_windowed_score(config)
    # create_csv_for_debugger(config)
    # cut_score_for_valid(config)

    # evaluate_validation_prediction(config)

    predict_for_test(config)
    aggregate_prediction_for_test(config)
    cut_score_for_test(config)

    # plot_tfpn_examples(config, 18)
    # list_confusion(config)

    # final_corrections(config)

    # compare_with_best_submssion(config)

    # show_score_details(config)


np.save
np.load
