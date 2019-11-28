def db_seresnext54_LSEP_VD():

    import torch
    from torch import nn

    model_name = 'db_seresnext54_LSEP_VD'

    device = 'cuda:0'

    torch.backends.cudnn.benchmark = True

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.utils import shuffle

    from skimage.transform import resize
    from skimage.color import rgb2gray, gray2rgb

    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    from tqdm import tqdm_notebook

    import gc
    import math
    import sys

    from fastai import *
    from fastai.vision import *

    np.random.seed(42)

    data_dir = '../../../input/'
    submit_l1_dir = "../../../submits/"
    weights_dir = "../../weights/"
    results_dir = '../../../results/'

    name_label_dict = {
    0:  'Nucleoplasm',
    1:  'Nuclear membrane',
    2:  'Nucleoli',
    3:  'Nucleoli fibrillar center',
    4:  'Nuclear speckles',
    5:  'Nuclear bodies',
    6:  'Endoplasmic reticulum',
    7:  'Golgi apparatus',
    8:  'Peroxisomes',
    9:  'Endosomes',
    10:  'Lysosomes',
    11:  'Intermediate filaments',
    12:  'Actin filaments',
    13:  'Focal adhesion sites',
    14:  'Microtubules',
    15:  'Microtubule ends',
    16:  'Cytokinetic bridge',
    17:  'Mitotic spindle',
    18:  'Microtubule organizing center',
    19:  'Centrosome',
    20:  'Lipid droplets',
    21:  'Plasma membrane',
    22:  'Cell junctions',
    23:  'Mitochondria',
    24:  'Aggresome',
    25:  'Cytosol',
    26:  'Cytoplasmic bodies',
    27:  'Rods & rings' }

    def kfold_threshold(y_true, y_pred):
        n_classes = len(name_label_dict)
        classes_thresholds = []
        classes_scores = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=239)
        for i in range(n_classes):
            kf_class_thresholds = []
            if (sum(y_true[:,i]) > 20):
                for _, tst_inx in kf.split(y_true,y_true[:,i]):
                    t_min = np.min(y_pred[tst_inx,i])
                    t_max = np.max(y_pred[tst_inx,i])
                    thresholds = np.linspace(t_min, t_max, 50)
                    scores = np.array([
                        f1_score(y_true[tst_inx,i], np.int32(y_pred[tst_inx,i] >= threshold)) for threshold in thresholds
                    ])
                    threshold_best_index = np.argmax(scores)
                    kf_class_thresholds.append(thresholds[threshold_best_index])
                threshold = np.mean(kf_class_thresholds)
                classes_thresholds.append(threshold)
                f1 = f1_score(y_true[:,i], np.int32(y_pred[:,i] >= threshold))
                classes_scores.append(f1)
            else:
                t_min = np.min(y_pred[:,i])
                t_max = np.max(y_pred[:,i])
                thresholds = np.linspace(t_min, t_max, 50)
                scores = np.array([
                    f1_score(y_true[:,i], np.int32(y_pred[:,i] >= threshold)) for threshold in thresholds
                ])
                threshold_best_index = np.argmax(scores)
                classes_thresholds.append(thresholds[threshold_best_index])
                f1 = f1_score(y_true[:,i], np.int32(y_pred[:,i] >= thresholds[threshold_best_index]))
                classes_scores.append(f1)
        return classes_thresholds, classes_scores

    import pretrainedmodels
    import pretrainedmodels.utils as pqutils

    _model_name = 'se_resnext50_32x4d'
    model = pretrainedmodels.__dict__[_model_name](num_classes=1000, pretrained='imagenet')
    tf_img = pqutils.TransformImage(model)
    model_stats = (tf_img.__dict__['mean'], tf_img.__dict__['std'])
    data_dir = '../../../input/'
    valid_df = pd.read_csv('../../../input/' + 'val_id.csv', header=None, names=['idx','Id'])
    train_df = pd.read_csv(data_dir + 'train.csv')
    test_matches = pd.read_csv('../../../input/test_matches.csv')
    test_matches.Extra = test_matches['Extra'].apply(lambda x : "_".join(x.split("_")[2:]))
    xtra_data = pd.read_csv(data_dir + 'xtra_train.csv')
    xtra_data['Extra'] = xtra_data.Id.apply(lambda x : x[:x.find("_classes")])

    xtra_matches_ids = test_matches.Extra.values.tolist()
    xtra_data_train = xtra_data.loc[~xtra_data.Extra.isin(xtra_matches_ids),['Id','Target']].reset_index(drop=True)
    xtra_data_valid = xtra_data.loc[xtra_data.Extra.isin(xtra_matches_ids),['Id','Target']].reset_index(drop=True)
    data = xtra_data_train
    labels = np.zeros((data.shape[0], 28), dtype=np.int32)
    if "Target" in data:
        for i, lbls in data['Target'].str.split().iteritems():
            for j in map(int, lbls):
                labels[i, j] = 1

    xtra_matches_ids = ['1054_E4_1_classes_25_16_0','1762_G4_5_classes_27','1335_C6_2_classes_3',
                        '935_D5_2_classes_22_0','27_H9_2_classes_10','669_D8_1_classes_16_2',
                        '1178_D4_2_classes_19_16_14','791_A9_1_classes_10_9','759_F9_9_classes_25_21_19_16',
                        '1283_F10_2_classes_16_0','688_E7_10_classes_23','1772_F9_7_classes_25_17',
                        '454_E5_1_classes_14_0','1020_C5_3_classes_23','1386_G4_2_classes_8',
                        '681_G8_5_classes_13','1609_C4_2_classes_16_0','690_D3_5_classes_22_21_1_0',
                        '1245_B2_2_classes_21_0','1335_C10_4_classes_16_0','693_A11_3_classes_23',
                        '1139_A12_4_classes_23','916_F8_1_classes_25_2_0','694_C1_2_classes_18_1',
                        '929_B8_1_classes_25_19','340_F5_3_classes_13','138_B12_1_classes_8',
                        '932_G11_2_classes_25_16','28_H9_1_classes_10','924_F12_1_classes_27',
                        '682_F12_2_classes_25_4','1147_D3_13_classes_16_0','346_A5_1_classes_12',
                        '616_F1_4_classes_8','73_A10_1_classes_27_25','663_A9_2_classes_16_14',
                        '859_C8_4_classes_16_14','933_C10_4_classes_22_21','1207_B10_7_classes_12',
                        '694_F10_1_classes_25_21','908_E3_1_classes_4','1758_C9_4_classes_17_2',
                        '1335_D2_2_classes_2_0','929_H2_2_classes_23','1717_G8_34_classes_25_17',
                        '1150_H4_7_classes_13','1054_E4_2_classes_25_16_0','504_B1_3_classes_25_16_0',
                        '747_B5_4_classes_10_9','1020_B1_7_classes_23_5','918_H10_2_classes_25_15',
                        '532_H3_1_classes_25_16_0','757_C6_3_classes_16_2','1346_H6_3_classes_16_5_0',
                        '496_D1_1_classes_16_0','1042_C3_3_classes_27','929_B12_1_classes_3',
                        '684_C4_2_classes_23_0','696_C9_5_classes_25_21_0','1144_A10_4_classes_2','846_A8_2_classes_16_14','903_F12_2_classes_23_5','1264_G1_1_classes_27','925_H8_2_classes_1_0','121_C6_2_classes_10_9','1657_E10_3_classes_25_17','932_G11_1_classes_25_16','704_G4_1_classes_25_12','1039_C3_2_classes_19_16','906_H7_2_classes_25_6','19_H7_2_classes_8','725_G10_2_classes_16_14','681_B2_4_classes_4','697_A6_4_classes_19_0','1581_B12_2_classes_16_14','926_F7_2_classes_5_0','1770_D2_4_classes_21_17_4','1037_F4_3_classes_19','1413_F11_6_classes_21_16','694_A2_1_classes_2','1049_D11_2_classes_25_16_0','1276_C3_2_classes_21_0','346_B12_3_classes_14_0','1773_G12_3_classes_16_12','1183_F4_2_classes_15','1158_H11_8_classes_16_5','380_C6_1_classes_16_0','792_B6_7_classes_13_0','682_C9_6_classes_25_12_2','906_A9_4_classes_20_0','400_D3_2_classes_25_7','1237_G1_4_classes_21_6','793_B1_1_classes_25_22_0','1308_A5_4_classes_5','800_E1_1_classes_16_14','1421_G5_7_classes_17','906_A9_6_classes_20_0','1245_B2_3_classes_21_0','626_D7_6_classes_25_21_12','344_G2_4_classes_11','901_E12_1_classes_25_6_2','1050_F6_6_classes_16_0','240_G8_1_classes_8','933_C2_1_classes_23_2_0','556_B9_1_classes_25_18_0','1335_C10_2_classes_16_0','1125_F6_3_classes_4','1495_F7_3_classes_7_0','694_C1_1_classes_18_1','918_B3_4_classes_14','1762_E6_5_classes_7','915_C6_5_classes_4','820_G4_3_classes_10_9','927_F12_12_classes_18_0','901_D10_2_classes_12_0','1642_G7_34_classes_25_16','928_G1_2_classes_14_7','682_G9_1_classes_7_0','903_F2_1_classes_2_0','1645_E1_32_classes_16_14','685_G10_5_classes_12_0','927_A9_10_classes_25_5','957_G6_4_classes_16','757_C6_2_classes_16_2','1213_C4_2_classes_4','909_A6_1_classes_2','694_D6_2_classes_1_0','480_D6_3_classes_25_16','1050_F1_3_classes_25_16_0','692_A1_5_classes_25_14_0','1772_H1_5_classes_18_17_16_0','991_G6_7_classes_10_9','782_F8_2_classes_25_16','693_H4_1_classes_7','1259_A11_4_classes_19_16','1414_D12_2_classes_21_0','1139_D5_5_classes_5','930_H3_2_classes_1','901_G9_5_classes_25_19_0','1754_G2_34_classes_5','353_A9_1_classes_21_13','1179_H7_1_classes_25_16_0','1423_A4_2_classes_16_14','686_F4_2_classes_22_21','1693_E1_2_classes_23_16','400_H8_2_classes_23','1680_G4_4_classes_16','935_G3_1_classes_5','838_E8_1_classes_3','1030_D8_2_classes_7_0','684_D12_4_classes_18','812_C10_2_classes_13_0','1416_D10_6_classes_21_16_0','1293_E3_2_classes_1_0','480_D6_2_classes_25_16','700_H6_2_classes_25_2','1773_E10_4_classes_16_0','611_E10_1_classes_25_13','346_B12_4_classes_14_0','523_A9_4_classes_5','1581_B12_3_classes_16_14','684_D8_6_classes_25_12_0','927_F12_11_classes_18_0','353_E4_2_classes_5','556_C1_5_classes_25_22_16','1179_H7_2_classes_25_16_0','1711_B12_3_classes_26_21_4','449_G8_2_classes_4_2','544_A8_5_classes_22_21_7','1772_H1_3_classes_18_17_16_0','1772_G2_6_classes_25_19_16_0','909_C11_2_classes_2_0','930_C12_1_classes_18_14_6','690_C10_2_classes_13','1009_B6_2_classes_10_9','757_E10_5_classes_12','88_D7_2_classes_8','383_E8_7_classes_25_17','1432_F2_2_classes_6','505_C10_1_classes_25_15','1104_E7_2_classes_16_14','699_E8_1_classes_1','1213_C4_3_classes_4','690_H5_1_classes_4','1169_D3_6_classes_16_0','686_F4_1_classes_22_21','532_D1_1_classes_16_0','896_G8_3_classes_5_0','934_G4_3_classes_21','344_G2_1_classes_11','369_C9_1_classes_18_14_0','682_F12_1_classes_25_4','683_E1_2_classes_25_1_0','697_G3_6_classes_13_7','1772_A6_7_classes_5','933_C4_6_classes_5','1231_F9_5_classes_7','802_D5_9_classes_16_0','682_G10_1_classes_7','850_C1_9_classes_21_0','929_B12_2_classes_3','1339_D3_3_classes_2_1','858_D4_2_classes_4','334_B12_2_classes_4','622_F1_7_classes_8','908_G5_2_classes_2_0','778_G6_2_classes_25_16_14','1027_C4_1_classes_7','886_C10_5_classes_23_0','807_C2_3_classes_4','1314_D2_2_classes_25_16_0','1770_B5_1_classes_21_16_11','1105_F10_2_classes_16_0','1283_B2_10_classes_16_0','583_E11_1_classes_25_16','820_G4_7_classes_10_9','928_H3_2_classes_14_0','970_H1_4_classes_25_18','1751_A7_32_classes_27','701_H10_2_classes_25_14','1773_B6_11_classes_23_17_16','1736_G7_31_classes_25_16','928_H3_1_classes_14_0','1645_E5_34_classes_17','539_B3_1_classes_25_21_0','683_E1_1_classes_25_1_0','484_G6_3_classes_22','928_A1_1_classes_4','1773_B6_7_classes_23_17_16','1255_A3_4_classes_16_0','698_C6_2_classes_25_21_4','1773_D5_6_classes_17','681_G8_4_classes_13','935_H11_2_classes_22_0','1125_B9_4_classes_25_7','698_F11_1_classes_13_0','344_F7_1_classes_25_21','906_C11_1_classes_4','1656_F5_2_classes_19_17','1761_A10_3_classes_23_17_14','1772_H5_7_classes_17_7','910_B8_1_classes_12_0','1283_F10_4_classes_16_0','508_C10_1_classes_25_15','681_B2_3_classes_4','868_E8_2_classes_17_16_0','1339_B9_2_classes_16_0','856_A2_4_classes_2_0','700_C3_6_classes_21','869_B3_1_classes_16_0','701_B9_2_classes_21_13_0','1178_F9_6_classes_16_0','542_G1_1_classes_11_2_0']


    xtra_data_train = xtra_data.loc[~xtra_data.Id.isin(xtra_matches_ids),['Id','Target']].reset_index(drop=True)
    xtra_data_valid = xtra_data.loc[xtra_data.Id.isin(xtra_matches_ids),['Id','Target']].reset_index(drop=True)

    exclude_valid = \
    ['5ae3db3a-bbc4-11e8-b2bc-ac1f6b6435d0',
     'e6d0b648-bbbc-11e8-b2ba-ac1f6b6435d0',
     '3202385a-bbca-11e8-b2bc-ac1f6b6435d0',
     '0cf36c82-bbca-11e8-b2bc-ac1f6b6435d0',
     '7cb0006e-bbaf-11e8-b2ba-ac1f6b6435d0',
     '87b77dd2-bba2-11e8-b2b9-ac1f6b6435d0',
     '62c88efa-bbc8-11e8-b2bc-ac1f6b6435d0',
     '44d819c2-bbbb-11e8-b2ba-ac1f6b6435d0',
     'b1ca2b40-bbbd-11e8-b2ba-ac1f6b6435d0',
     '8cd67266-bbbe-11e8-b2ba-ac1f6b6435d0',
     'cead83ec-bb9a-11e8-b2b9-ac1f6b6435d0',
     'a166d11a-bbca-11e8-b2bc-ac1f6b6435d0',
     '91a0a67e-bb9e-11e8-b2b9-ac1f6b6435d0',
     '2be24582-bbb1-11e8-b2ba-ac1f6b6435d0']
    exclude_train = \
    ['7138c4aa-bb9b-11e8-b2b9-ac1f6b6435d0',
     '8a10533e-bba6-11e8-b2ba-ac1f6b6435d0',
     'be92e108-bbb5-11e8-b2ba-ac1f6b6435d0',
     'abfa727e-bba4-11e8-b2ba-ac1f6b6435d0',
     '2384acac-bbae-11e8-b2ba-ac1f6b6435d0',
     'c7a7a462-bbb1-11e8-b2ba-ac1f6b6435d0',
     '559f7ce0-bbb2-11e8-b2ba-ac1f6b6435d0']

    valid_df = pd.read_csv('../../../input/' + 'val_id.csv', header=None, names=['idx','Id'])
    valid_df = valid_df.loc[~valid_df.Id.isin(exclude_valid),:]
    train_df = pd.read_csv(data_dir + 'train.csv')
    train_df = train_df.loc[~train_df.Id.isin(exclude_train),:]

    test_df = pd.read_csv('../../../input/' + "sample_submission.csv")
    train = train_df.loc[~train_df.Id.isin(valid_df.Id.values.tolist()),:].reset_index(drop=True)
    train = pd.concat([train,xtra_data_train], axis=0, sort=False)
    valid = train_df.loc[train_df.Id.isin(valid_df.Id.values.tolist()),:].reset_index(drop=True)
    valid = pd.concat([valid,xtra_data_valid], axis=0, sort=False)
    test  = test_df
    del train_df,valid_df,test_df,xtra_data_valid,xtra_data_train
    gc.collect()

    train_files = train.Id.apply(lambda s: '../../../input/' + 'train_png/'+s+'.png')
    train_labels = train.Target.astype(str).apply(lambda s : [name_label_dict[int(q)] for q in s.split(' ')])
    train_ds = ImageMultiDataset(fns = train_files, labels = train_labels, classes = list(name_label_dict.values()))
    del train_files, train_labels

    valid_files = valid.Id.apply(lambda s: '../../../input/' + 'train_png/'+s+'.png')
    valid_labels = valid.Target.astype(str).apply(lambda s : [name_label_dict[int(q)] for q in s.split(' ')])
    valid_ds = ImageMultiDataset(fns = valid_files, labels = valid_labels, classes = list(name_label_dict.values()))
    del valid_files, valid_labels

    test_files = test.Id.apply(lambda s: '../../../input/' + 'test_png/'+s+'.png')
    test_labels = test.Predicted.astype(str).apply(lambda s : [name_label_dict[int(q)] for q in s.split(' ')])
    test_ds = ImageMultiDataset(fns = test_files, labels = test_labels, classes = list(name_label_dict.values()))
    del test_files, test_labels

    magnitudes = torch.tensor(np.zeros(8))
    magnitudes[6]=0.5
    xtra = [RandTransform(squish, {}),jitter(magnitude=0.03,p=0.1),perspective_warp(magnitude=0.03,p=0.1)]
    tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0,
                          max_zoom=1.2, max_lighting=0.2, max_warp=None,
                          p_affine=0.9, p_lighting=0.65, xtra_tfms=xtra)
    data = ImageDataBunch.create(train_ds, valid_ds, test_ds, path=data_dir, device=device,
                                 size=512, bs=12, ds_tfms=tfms, padding_mode='zeros')
    data.normalize(model_stats)

    from torch.autograd import Variable, Function

    def _to_one_hot(y, n_dims, dtype=torch.cuda.FloatTensor):
        scatter_dim = len(y.size())
        y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), n_dims).type(dtype)

        return zeros.scatter(scatter_dim, y_tensor, 1)

    class LSEP2(Function):
        """
        Autograd function of LSEP loss. Appropirate for multi-label
        - Reference: Li+2017
          https://arxiv.org/pdf/1704.03135.pdf
          Code-ref: https://github.com/Mipanox/Bird_cocktail/blob/196e9404a4f7022d1e56433112f581b34a334e53/utils.py#L332
        """

        @staticmethod
        def forward(ctx, input, target):
            batch_size = target.size()[0]
            label_size = target.size()[1]

            ##
            positive_indices = target.gt(0).float()
            negative_indices = target.eq(0).float()

            ## summing over all negatives and positives
            loss = 0.
            for i in range(input.size()[0]):
                pos = positive_indices[i].nonzero()
                neg = negative_indices[i].nonzero()
                pos_examples = input[i, pos]
                neg_examples = torch.transpose(input[i, neg], 0, 1)
                loss += torch.sum(torch.exp(neg_examples - pos_examples))

            loss = torch.log(1 + loss)

            ctx.save_for_backward(input, target)
            ctx.loss = loss
            ctx.positive_indices = positive_indices
            ctx.negative_indices = negative_indices

            return loss

        # This function has only a single output, so it gets only one gradient
        @staticmethod
        def backward(ctx, grad_output):
            dtype = torch.cuda.FloatTensor
            input, target = ctx.saved_variables
            N = input.size()[1]
            loss = Variable(ctx.loss, requires_grad = False)
            positive_indices = ctx.positive_indices
            negative_indices = ctx.negative_indices

            fac  = -1 / loss
            grad_input = torch.zeros(input.size()).type(dtype)

            scale = grad_input.size(0), -1
            phot = _to_one_hot(positive_indices.nonzero()[:, 1].view(*scale), N)
            nhot = _to_one_hot(negative_indices.nonzero()[:, 1].view(*scale), N)

            scale = (len(phot), *nhot.size())
            diffs = torch.sum(phot - nhot.expand(scale), dim=2)
            grads_input = (Variable(diffs * torch.exp(-input * diffs)) * (grad_output * fac))

            return grad_input, None, None

        class FocalLoss(nn.Module):
            def __init__(self, gamma=2):
                super().__init__()
                self.gamma = gamma

            def forward(self, input, target):
                if not (target.size() == input.size()):
                    raise ValueError("Target size ({}) must be the same as input size ({})"
                                     .format(target.size(), input.size()))

                max_val = (-input).clamp(min=0)
                loss = input - input * target + max_val + \
                    ((-max_val).exp() + (-input - max_val).exp()).log()

                invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
                loss = (invprobs * self.gamma).exp() * loss

                return loss.sum(dim=1).mean()
    from fastai.vision import models
    from sklearn import metrics
    import torchvision

    q = None

    class F1Metric(Callback):
        def on_epoch_begin(self, **kwargs):
            self.tp, self.sum_true, self.sum_pred = torch.zeros([28]).cuda(device),\
    torch.zeros([28]).cuda(device), torch.zeros([28]).cuda(device)
            self.threshold = 0.5
            self.metric = 0

        def on_batch_end(self, last_output, last_target, **kwargs):
            y_pred = torch.ge(last_output, self.threshold).float()
            y_true = last_target.float()

            self.tp = self.tp + (y_pred * y_true).sum(dim=0)
            self.sum_true = self.sum_true + (y_true).sum(dim=0)
            self.sum_pred = self.sum_pred + (y_pred).sum(dim=0)

        def on_epoch_end(self, **kwargs):
            global q
            eps = 1e-9
            q = self.tp
            true_positive = self.tp
            precision = true_positive.div(self.sum_true.add(eps))
            recall = true_positive.div(self.sum_pred.add(eps))

            self.metric = ((precision*recall).div(precision + recall + eps).mul(2)).mean()

    f1metric = F1Metric()

    def body(pretrained = True):
        return pretrainedmodels.__dict__[_model_name](num_classes=1000, pretrained='imagenet').to(device)

    learner = create_cnn(data, arch=body, cut=-2,
                         custom_head = create_head(4096, len(data.classes), ps=0.3), metrics=[f1metric]) #
    learner.loss_fn = FocalLoss()

    import collections


    layers = []

    for i, l1 in enumerate(learner.model):
        for j, l2 in enumerate(learner.model[i]):
            if isinstance(learner.model[i][j], collections.Iterable):
                for k, l3 in enumerate(learner.model[i][j]):
                    print(l3)
                    layers.append(learner.model[i][j][k])
            else:
                    layers.append(learner.model[i][j])

    learner.split(layers)


    learner.load('pd_seresnext_v3_0.7062.pnt')

    num_iter=0
    def get_lrs2(lr):
        global num_iter
        num_iter=num_iter+17
        np.random.seed(num_iter)
        return np.logspace(np.log(lr/16), np.log(lr), len(learner.layer_groups), base=np.e)+\
                    np.random.normal(0,lr*1e-2,size=len(learner.layer_groups))

    learner.loss_fn = LSEP2()

    lr = 3e-5
    learner.fit_one_cycle(4, max_lr=get_lrs2(lr))

    y_pred_solo, avg_preds1, y = learner.TTA(beta=None)
    y = y.cpu().numpy().copy()
    _, avg_preds2, _ = learner.TTA(beta=None)
    _, avg_preds3, _ = learner.TTA(beta=None)
    _, avg_preds4, _ = learner.TTA(beta=None)

    avg_preds = y_pred_solo.cpu().numpy().copy()*0.4+\
    torch.stack([avg_preds1,avg_preds2,avg_preds3,avg_preds4]).mean(0).cpu().numpy().copy()*0.6

    results_dir = '../../../results/'
    np.save(results_dir+model_name+'_y.npy', y)
    np.save(results_dir+model_name+'_ids.npy', valid.Id.values)
    np.save(results_dir+model_name+'_holdout.npy', avg_preds)