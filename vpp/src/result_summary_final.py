# -*- coding: UTF-8 -*-
import os
import numpy as np
from utils.multi_thres_and_leak import use_threshold, replace_leak_write_result


def summary_scores(score_files, save_path, weight=None, save_result=True):
    print'total {} result'.format(len(score_files)),
    if weight is None:
        weight = [1 for _ in xrange(len(score_files))]
    assert len(score_files) == len(weight), 'Error length of score_files not queal to weight'
    scores = []
    for i, sub_file in enumerate(score_files):
        scores.append(np.load(sub_file) * weight[i])
    scores = np.array(scores)
    ave_scores = np.sum(scores, 0) / sum(weight)
    if save_result:
        np.save(save_path, ave_scores)
        print 'save to:', save_path


def summary_scores_lcp_inceptionv3_800():
    score_files = [
        # inceptionv3  800
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_0_epoch8_score.npy',
        # 0.582
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_0_epoch18_score.npy',
        # 0.578
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_1_epoch12_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_2_epoch15_score.npy',
        # 0.580
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_3_epoch21_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_4_epoch14_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_4_epoch14_score.npy',
        # 0.589
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_5_epoch13_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_6_epoch12_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_7_epoch12_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_8_epoch12_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_9_epoch12_score.npy',
    ]
    save_path = './results/lcp_inceptionv3_800_score.npy'
    summary_scores(score_files, save_path)
    multi_thres_file = use_threshold(save_path)


def summary_scores_xie_inceptionv3_800():
    score_files = [
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_3_epoch28_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_3_epoch9_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_1/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_1_3_epoch12_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_2/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_2_3_epoch12_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_3/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_3_3_epoch12_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_4/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_4_3_epoch12_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_3_epoch23_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_3_epoch13_score.npy',
    ]
    save_path = './results/xie_inceptionv3_800_score.npy'
    summary_scores(score_files, save_path)
    multi_thres_file = use_threshold(save_path)


def summary_scores_inceptionv3_800():
    summary_scores_lcp_inceptionv3_800()
    summary_scores_xie_inceptionv3_800()
    score_files = [
        './results/lcp_inceptionv3_800_score.npy',
        './results/xie_inceptionv3_800_score.npy']
    save_path = './results/inceptionv3_800_score.npy'
    summary_scores(score_files, save_path, weight=[21, 10])
    multi_thres_file = use_threshold(save_path)


def summary_scores_inceptionv3_650():
    score_files = [
        # inceptionv3 650
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold_0_epoch13_score.npy',
        './models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_5fold_pretrain/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_5fold_pretrain_3_epoch13_score.npy',
    ]
    save_path = './results/inceptionv3_650_score.npy'
    summary_scores(score_files, save_path)
    multi_thres_file = use_threshold(save_path)


def summary_scores_inceptionv4_800():
    score_files = [
        # inceptionv4 800
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_0_epoch13_score.npy',
        # 0.589
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_1_epoch12_score.npy',
        # 0.587
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_2_epoch12_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_2_epoch15_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_3_epoch12_score.npy',
        # 0.589
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_3_epoch16_score.npy',
        # 0.589
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_4_epoch12_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_4_epoch15_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_5_epoch12_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_6_epoch16_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_7_epoch14_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_7_epoch20_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_8_epoch12_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_8_epoch22_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_9_epoch13_score.npy',
    ]
    save_path = './results/inceptionv4_800_score.npy'
    summary_scores(score_files, save_path)
    multi_thres_file = use_threshold(save_path)


def summary_scores_inceptionv4_650():
    score_files = [
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold_0_epoch17_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold_1_epoch14_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold_2_epoch13_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold_3_epoch16_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold_4_epoch15_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold_5_epoch13_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold_6_epoch13_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold_7_epoch14_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold_8_epoch16_score.npy',
    ]
    save_path = './results/inceptionv4_650_score.npy'
    summary_scores(score_files, save_path)
    multi_thres_file = use_threshold(save_path)


def summary_scores_xception_800():
    score_files = [
        # xception 800
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_0_epoch14_score.npy',
        # 0.577
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_1_epoch18_score.npy',
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_2_epoch18_score.npy',
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_3_epoch18_score.npy',
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_4_epoch14_score.npy',
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_5_epoch17_score.npy',
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_6_epoch13_score.npy',
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_7_epoch13_score.npy',
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_8_epoch13_score.npy',
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_9_epoch12_score.npy',
    ]
    save_path = './results/xception_800_score.npy'
    summary_scores(score_files, save_path)
    multi_thres_file = use_threshold(save_path)


def summary_scores_xception_650():
    score_files = [
        # xception 650
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold_0_epoch8_score.npy',
        # 0.592
    ]
    save_path = './results/xception_650_score.npy'
    summary_scores(score_files, save_path)
    multi_thres_file = use_threshold(save_path)


def summary_scores_xception_512():
    score_files = [
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_5fold_0_epoch17_score.npy',
        # 0.569
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_5fold_1_epoch12_score.npy',
        # 0.565
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_5fold_3_epoch14_score.npy',
        # 0.577
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_0_epoch22_score.npy',
        # 0.572
        './models/xception_fc/xception_fc_offi_lr0.001_weightedsamper_mlsm_rms_lrexp_pretrain/submit/xception_fc_offi_lr0.001_weightedsamper_mlsm_rms_lrexp_pretrain_0_epoch6_score.npy',
        # 0.574
    ]
    save_path = './results/xception_512_score.npy'
    summary_scores(score_files, save_path)
    multi_thres_file = use_threshold(save_path)


def feature_concat_incv3_incv4_xcep_fc():
    # base_path = '/disk/223/xiejb231/unet_mxnet2ncnn/datasets/humanprotein/scripts/Ensemble/FC/inceptionv3v4x_fc_800_offi_hpa/lr0.05_exp_mlsm_sgd_1layer_dropout_2layer/submit/inceptionv3v4x_fc_800_offi_hpa_submission_fold'
    base_path = './models/MLP/lr0.5_step_0.5_10_mlsm_sgd_2layer_10fold/submit/inceptionv3_inceptionv4_xception_800_offi_hpa_submission_fold'
    score_files = [
        base_path + '0_score.npy',
        base_path + '1_score.npy',
        base_path + '2_score.npy',
        base_path + '3_score.npy',
        base_path + '4_score.npy',
        base_path + '5_score.npy',
        base_path + '6_score.npy',
        base_path + '7_score.npy',
        base_path + '8_score.npy',
        base_path + '9_score.npy',
    ]
    save_path = './results/featureconcat_incv3_800_incv4_800_xcep_800_score.npy'
    summary_scores(score_files, save_path)
    multi_thres_file = use_threshold(save_path)


def feature_concat_incv3_incv4_xcep_fc_2():
    # base_path = '/disk/223/xiejb231/unet_mxnet2ncnn/datasets/humanprotein/scripts/Ensemble/FC/inceptionv3v4x_fc_800_offi_hpa/lr0.05_exp_mlsm_sgd_1layer_dropout_2layer/submit/inceptionv3v4x_fc_800_offi_hpa_submission_fold'
    base_path = './models/MLP/lr0.5_exp_bce_sgd_2layer_10fold/submit/inceptionv3_inceptionv4_xception_800_offi_hpa_submission_fold'
    score_files = [
        base_path + '0_score.npy',
        base_path + '1_score.npy',
        base_path + '2_score.npy',
        base_path + '3_score.npy',
        base_path + '4_score.npy',
        base_path + '5_score.npy',
        base_path + '6_score.npy',
        base_path + '7_score.npy',
        base_path + '8_score.npy',
        base_path + '9_score.npy',
    ]
    save_path = './results/featureconcat_incv3_800_incv4_800_xcep_800_2_score.npy'
    summary_scores(score_files, save_path)
    multi_thres_file = use_threshold(save_path)


def summary_scores_any_sub():
    score_files_weight = [
        ['./results/inceptionv3_800_score.npy', 31],
        ['./results/inceptionv3_650_score.npy', 11],
        ['./results/inceptionv4_800_score.npy', 21],
        ['./results/inceptionv4_650_score.npy', 6],
        ['./results/xception_800_score.npy', 9],
        ['./results/xception_650_score.npy', 2],
        ['./results/xception_512_score.npy', 15],
        ['./results/featureconcat_incv3_800_incv4_800_xcep_800_score.npy', 13],  # 0.630
        ['./results/featureconcat_incv3_800_incv4_800_xcep_800_2_score.npy', 13]
    ]

    score_files = [_[0] for _ in score_files_weight]
    weight = [_[1] for _ in score_files_weight]

    for _it in xrange(len(score_files)):
        print os.path.basename(score_files[_it]), ':', weight[_it]

    print weight
    save_path = './results/summary_final_1_score.npy'
    summary_scores(score_files, save_path, weight)
    multi_thres_file = use_threshold(save_path)
    final_commit_file = replace_leak_write_result(multi_thres_file, show_replace=False)
    print '*' * 84
    print '*    It\'s our first final submission --> ' + final_commit_file + '    *'
    print '*' * 84


def summary_scores_any_sub_2():
    score_files_weight = [
        [
            '/disk/223/lichuanpeng/Project_Models/Kaggle/HumanProtein/result_summary/best_submit/summary_sub_646_score.npy',
            1],
        ['/disk/223/lichuanpeng/Project_Models/Kaggle/HumanProtein/result_summary/best_submit/xie_646.npy', 1],
        ['/disk/223/lichuanpeng/Project_Models/Kaggle/HumanProtein/result_summary/best_submit/sub_9_645.npy', 1],
    ]
    score_files = [_[0] for _ in score_files_weight]
    weight = [_[1] for _ in score_files_weight]

    for _it in xrange(len(score_files)):
        print os.path.basename(score_files[_it]), ':', weight[_it]

    save_path = './results/summary_final_2_score.npy'
    summary_scores(score_files, save_path, weight)
    multi_thres_file = use_threshold(save_path)
    final_commit_file = replace_leak_write_result(multi_thres_file, show_replace=False)
    print '*' * 85
    print '*    It\'s our first final submission --> ' + final_commit_file + '    *'
    print '*' * 85

if __name__ == '__main__':
    summary_scores_inceptionv3_800()
    summary_scores_inceptionv3_650()
    summary_scores_inceptionv4_800()
    summary_scores_inceptionv4_650()
    summary_scores_xception_512()
    summary_scores_xception_800()
    summary_scores_xception_650()
    feature_concat_incv3_incv4_xcep_fc()
    feature_concat_incv3_incv4_xcep_fc_2()
    summary_scores_any_sub()
    summary_scores_any_sub_2()

