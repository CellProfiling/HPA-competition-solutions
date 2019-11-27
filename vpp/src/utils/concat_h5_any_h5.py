# -*- coding: UTF-8 -*-
import os
import numpy as np
import tqdm
from feature_utils import FeatureDataSet
from sklearn import preprocessing


def concat_feature_by_h5(h5_files, save_to, sub_feature_with_l2=True, concat_feature_with_l2=True):
    """

    :param h5_files: list, absolute path of h5
    :param save_to: path to save ex:/xx/xx/xxx.h5
    :return:
    """
    assert isinstance(h5_files, list), 'Error: type of h5_files must be list！'
    assert len(h5_files) > 1, 'Error: h5_files(length:{}) too short！'.format(len(h5_files))
    assert not os.path.exists(save_to), 'Error: create h5({}) failed, This file already exists!'.format(save_to)
    h5_features = []
    for sub_file in h5_files:
        assert os.path.exists(sub_file), 'Error: No Such File:{}'.format(sub_file)
        h5_features.append(FeatureDataSet(sub_file, 0, readonly=True))

    BATCH_SIZE = 1024

    # check feature length
    for it in xrange(len(h5_features) - 1):
        assert h5_features[it].num_features() == h5_features[
            it + 1].num_features(), 'Error: length of index-({},{}) and index-({},{}) not identical'.format(it,
                                                                                                            h5_features[
                                                                                                                it].num_features(),
                                                                                                            it + 1,
                                                                                                            h5_features[
                                                                                                                it + 1].num_features())
    feature_length_concat = h5_features[0].num_features()

    scaler_l2 = preprocessing.Normalizer()

    # check feature dim
    feature_dim_concat = 0
    for it in xrange(len(h5_features)):
        feature_dim_concat += h5_features[it].feature_shape()[1]

    h5_save = FeatureDataSet(save_to, feature_dim_concat)
    pBar = tqdm.tqdm(total=feature_length_concat)
    for start in range(0, feature_length_concat, BATCH_SIZE):
        end = min(start + BATCH_SIZE, feature_length_concat)
        tmp_features = np.zeros((end - start, feature_dim_concat))
        flag_feature_dim_start = 0
        for it_h5 in h5_features:
            sub_feature = it_h5.readFeature(start, end)
            tmp_features[:, flag_feature_dim_start:flag_feature_dim_start + sub_feature.shape[
                1]] = sub_feature if not sub_feature_with_l2 else scaler_l2.fit_transform(sub_feature)
            flag_feature_dim_start = flag_feature_dim_start + sub_feature.shape[1]
        h5_save.writeFeature(tmp_features if not concat_feature_with_l2 else scaler_l2.fit_transform(tmp_features))
        pBar.update(end - start)
    pBar.close()
    h5_save.close()


def do_concat(h5_files, save_to):
    concat_feature_by_h5(h5_files, save_to, sub_feature_with_l2=True, concat_feature_with_l2=True)
    h5_read = FeatureDataSet(save_to, 0, readonly=True)
    print 'feature saved to :', save_to
    print 'shape:', h5_read.feature_shape()


if __name__ == '__main__':
    test_h5_files = [
        '../models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_0_epoch18_feature.h5',
        '../models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_1_epoch20_feature.h5',
        '../models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_2_epoch21_feature.h5',
        '../models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_3_epoch21_feature.h5',
        '../models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_4_epoch14_feature.h5',
        '../models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_5_epoch13_feature.h5',
        '../models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_6_epoch12_feature.h5',
        '../models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_7_epoch12_feature.h5',
        '../models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_8_epoch12_feature.h5',
        '../models/inceptionv3_fc/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv3_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_9_epoch12_feature.h5',
        '../models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_0_epoch13_feature.h5',
        '../models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_1_epoch12_feature.h5',
        '../models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_2_epoch15_feature.h5',
        '../models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_3_epoch12_feature.h5',
        '../models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_4_epoch15_feature.h5',
        '../models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_5_epoch12_feature.h5',
        '../models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_6_epoch16_feature.h5',
        '../models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_7_epoch20_feature.h5',
        '../models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_8_epoch22_feature.h5',
        '../models/inceptionv4_fc/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/inceptionv4_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_9_epoch13_feature.h5',
        '../models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_0_epoch14_feature.h5',
        '../models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_1_epoch18_feature.h5',
        '../models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_2_epoch18_feature.h5',
        '../models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_3_epoch12_feature.h5',
        '../models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_4_epoch14_feature.h5',
        '../models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_5_epoch12_feature.h5',
        '../models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_6_epoch13_feature.h5',
        '../models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_7_epoch13_feature.h5',
        '../models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_8_epoch13_feature.h5',
        '../models/xception_fc/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold/submit/xception_fc_offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold_9_epoch12_feature.h5',
    ]

    test_save_to = '../features/inceptionv3_inceptionv4_xception_800_10fold/test_offi_hpa_features.h5'

    do_concat(test_h5_files, test_save_to)

