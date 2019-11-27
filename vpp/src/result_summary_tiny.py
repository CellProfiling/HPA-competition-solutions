# -*- coding: UTF-8 -*-
import os
import numpy as np
import pandas as pd
from utils.multi_thres_and_leak import *

def use_threshold(result_npy_file):

    #It's the best threshold
    threshold=np.array([0.422, 0.15, 0.454,  0.29,  0.348, 0.331, 0.15,  0.572, 0.15,0.15,  0.15,  0.15,  0.15 , 0.15 , 0.15,  0.15,  0.299,  0.15 , 0.15 , 0.15 , 0.15 , 0.318 ,0.15  ,0.336, 0.15 , 0.355 ,0.15  ,0.15 ])

    print 'new Threshold',threshold

    sample_file = './data/sample_submission.csv'
    sample_submission_df = pd.read_csv(sample_file)

    result_scores = np.load(result_npy_file)

    assert len(sample_submission_df['Predicted']) == result_scores.shape[0], 'Error'

    submissions = []
    for it, row in enumerate(result_scores):
        sub_label = row-threshold
        sub_label = sub_label>0
        subrow = ' '.join(list([str(i) for i in np.nonzero(sub_label)[0]]))
        if len(np.nonzero(sub_label)[0]) == 0:
            arg_maxscore = np.argmax(result_scores[it])
            subrow = str(arg_maxscore)
        #print subrow
        submissions.append(subrow)
    # print subrow
    sample_submission_df['Predicted'] = submissions
    save_file = result_npy_file[:-10]+'_multhr.csv'
    sample_submission_df.to_csv(save_file, index=None)
    print '[multi-threshold]result save to ', save_file
    return save_file

def summary_scores(score_files,save_path,weight=None,save_result=True):
    print'total {} result'.format(len(score_files)),
    if weight is None:
        weight=[1 for _ in xrange(len(score_files))]
    assert len(score_files)==len(weight),'Error length of score_files not queal to weight'
    scores = []
    for i, sub_file in enumerate(score_files) :
        scores.append(np.load(sub_file)*weight[i])
    scores = np.array(scores)
    ave_scores = np.sum(scores, 0)/sum(weight)
    if save_result:
        np.save(save_path, ave_scores)
        print 'save to:',save_path


def summary_scores_inceptionv4_800():
    score_files = [
        # inceptionv4 800
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_1_epoch12_score.npy',# 0.587
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_2_epoch15_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_2_epoch15_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_3_epoch12_score.npy',# 0.589
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_5_epoch12_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_8_epoch22_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_8_epoch22_score.npy',
        './models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_9_epoch13_score.npy',
    ]
    save_path = './results/inceptionv4_800_tiny_score.npy'
    summary_scores(score_files, save_path)
    use_threshold(save_path)

def summary_scores_xception_800():
    score_files = [
        # xception 800
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_0_epoch14_score.npy', # 0.577
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_6_epoch13_score.npy',
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_7_epoch13_score.npy',
    ]
    save_path = './results/xception_800_tiny_score.npy'
    summary_scores(score_files, save_path)
    use_threshold(save_path)



def summary_scores_xception_512():
    score_files = [
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_5fold_0_epoch17_score.npy',
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_5fold_1_epoch12_score.npy',
        './models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_0_epoch22_score.npy',
        './models/xception_fc/xception_fc_offi_lr0.001_weightedsamper_mlsm_rms_lrexp_pretrain/submit/xception_fc_offi_lr0.001_weightedsamper_mlsm_rms_lrexp_pretrain_0_epoch6_score.npy',
       ]
    save_path = './results/xception_512_tiny_score.npy'
    summary_scores(score_files, save_path)
    use_threshold(save_path)


def summary_scores_any_sub():

    score_files_weight = [
        ['./models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_5_epoch13_score.npy', 0.5],
        ['./results/inceptionv4_800_tiny_score.npy',1.8],
        ['./results/xception_800_tiny_score.npy',   0.7],
        ['./results/xception_512_tiny_score.npy',   1.3],
    ]

    score_files = [_[0] for _ in score_files_weight]
    weight = [_[1] for _ in score_files_weight]

    for _it in xrange(len(score_files)):
        print os.path.basename(score_files[_it]),':',weight[_it]

    print weight
    save_path = './results/tiny_submission_score.npy'
    summary_scores(score_files, save_path,weight)
    multi_thres_file = use_threshold(save_path)
    final_commit_file = replace_leak_write_result(multi_thres_file, show_replace=False)

    print '*'*77
    print '*    It\'s our final submission --> '+final_commit_file+'    *'
    print '*'*77


if __name__ == '__main__':
    summary_scores_inceptionv4_800()
    summary_scores_xception_512()
    summary_scores_xception_800()
    summary_scores_any_sub()





