from fastai.conv_learner import *
from fastai.dataset import *
from tensorboard_cb_old import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scipy.optimize as opt
from sklearn.model_selection import StratifiedKFold
from itertools import chain
from collections import Counter
from random import shuffle
import pickle as pkl

import warnings

warnings.filterwarnings("ignore")

# =======================================================================================================================
# Something
# =======================================================================================================================

PATH = './'
# TRAIN = '../input/HPAv18/png/'
TRAIN = '../input/train/'
TEST = '../input/test/'

LABELS_org = '../input/train.csv'
LABELS_fld1 = '../input/train_fld1.csv'
LABELS_fld1_val = '../input/valid_fld1.csv'

LABELS_ext = '../input/HPAv18RBGY_wodpl.csv'
LABELS_all = '../input/train_org_ext.csv'

SAMPLE = '../input/sample_submission.csv'

name_label_dict = {
    0: 'Nucleoplasm',
    1: 'Nuclear membrane',
    2: 'Nucleoli',
    3: 'Nucleoli fibrillar center',
    4: 'Nuclear speckles',
    5: 'Nuclear bodies',
    6: 'Endoplasmic reticulum',
    7: 'Golgi apparatus',
    8: 'Peroxisomes',
    9: 'Endosomes',
    10: 'Lysosomes',
    11: 'Intermediate filaments',
    12: 'Actin filaments',
    13: 'Focal adhesion sites',
    14: 'Microtubules',
    15: 'Microtubule ends',
    16: 'Cytokinetic bridge',
    17: 'Mitotic spindle',
    18: 'Microtubule organizing center',
    19: 'Centrosome',
    20: 'Lipid droplets',
    21: 'Plasma membrane',
    22: 'Cell junctions',
    23: 'Mitochondria',
    24: 'Aggresome',
    25: 'Cytosol',
    26: 'Cytoplasmic bodies',
    27: 'Rods & rings'}

nw = 4  # number of workers for data loader
arch = resnet34  # specify target architecture

# =======================================================================================================================
# Data
# =======================================================================================================================
df_train_fld1 = pd.read_csv(LABELS_fld1)
df_val_fld1 = pd.read_csv(LABELS_fld1_val)
df_train_ext = pd.read_csv(LABELS_ext)

# remove broken images
df_train_fld1 = df_train_fld1[(df_train_fld1.Id != 'dc756dea-bbb4-11e8-b2ba-ac1f6b6435d0') &
                    (df_train_fld1.Id != 'c861eb54-bb9f-11e8-b2b9-ac1f6b6435d0') &
                    (df_train_fld1.Id != '7a88f200-bbc3-11e8-b2bc-ac1f6b6435d0')
                    ]
df_val_fld1 = df_val_fld1[(df_val_fld1.Id != 'dc756dea-bbb4-11e8-b2ba-ac1f6b6435d0') &
                    (df_val_fld1.Id != 'c861eb54-bb9f-11e8-b2b9-ac1f6b6435d0') &
                    (df_val_fld1.Id != '7a88f200-bbc3-11e8-b2bc-ac1f6b6435d0')
                    ]
df_train_ext = df_train_ext[df_train_ext.Id != '27751_219_G10_1']


print(df_train_fld1.shape[0], 'training masks')
print(df_val_fld1.shape[0], 'validation masks')
print(df_train_ext.shape[0], 'external masks')

# loading original data twice to balance external data
tr_n = df_train_fld1['Id'].values.tolist() + df_train_ext['Id'].values.tolist() + df_train_fld1['Id'].values.tolist() + df_train_fld1['Id'].values.tolist()
shuffle(tr_n)

val_n = df_val_fld1['Id'].values.tolist()
tr_n = tr_n[:-1]  # pytorch has problems if last batch has one sample

test_names = list({f[:36] for f in os.listdir(TEST)})
#=======================================================================================================================

def open_rgby(path, id):  # a function that reads RGBY image
    # print(id)
    #colors = ['red','green','blue','yellow']
    colors = ['red', 'green', 'blue']
    flags = cv2.IMREAD_GRAYSCALE
    try:
        img = [cv2.imread(os.path.join(path, id + '_' + color + '.png'), flags).astype(np.float32) / 255
               for color in colors]
        img = np.stack(img, axis=-1)
        return img
    except:
        print('img broken:', id)


class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.labels = pd.read_csv(LABELS_all).set_index('Id')
        self.labels['Target'] = [[int(i) for i in s.split()] for s in self.labels['Target']]
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        img = open_rgby(self.path, self.fnames[i])
        if self.sz == 512:
            return img
        else:
            return cv2.resize(img, (self.sz, self.sz), cv2.INTER_AREA)

    def get_y(self, i):
        if (self.path == TEST):
            return np.zeros(len(name_label_dict), dtype=np.int)
        else:
            labels = self.labels.loc[self.fnames[i]]['Target']
            return np.eye(len(name_label_dict), dtype=np.float)[labels].sum(axis=0)

    @property
    def is_multi(self):
        return True

    @property
    def is_reg(self):
        return True

    # this flag is set to remove the output sigmoid that allows log(sigmoid) optimization
    # of the numerical stability of the loss function

    def get_c(self):
        return len(name_label_dict)  # number of classes


def get_data(sz, bs):
    # data augmentation
    aug_tfms = [RandomRotate(30, tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO),
                # RandomCrop(480, tfm_y=TfmType.NO),
                # RandomRotateZoom(deg=30, zoom=1, stretch=0.5)
                ]
    # mean and std in of each channel in the train set
    # Original set: [0.0869  0.0599  0.06534 0.08923] [0.13047 0.09831 0.14875 0.1333 ]
    # Extra data:[0.04445 0.0499  0.01867 0.10086] [0.05364 0.07222 0.02486 0.13921]
    # combined: [0.05697 0.05282 0.03241 0.09748] [0.08622 0.08092 0.08611 0.13758]
    stats = A([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313])
    #stats = A([0.05697, 0.05282, 0.03241, 0.09748] , [0.08622, 0.08092, 0.08611, 0.13758])
    #stats = A([0.05697, 0.05282, 0.03241], [0.08622, 0.08092, 0.08611])

    tfms = tfms_from_stats(stats, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
                           aug_tfms=aug_tfms)
    ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n) % bs)], TRAIN),
                          (val_n, TRAIN), tfms, test=(test_names, TEST))
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    return md


# ======================================================================================================================
# Display
# =======================================================================================================================

# bs = 16
# sz = 256
# md = get_data(sz,bs)
#
# x,y = next(iter(md.trn_dl))
# print(x.shape, y.shape)
#
# def display_imgs(x):
#     columns = 4
#     bs = x.shape[0]
#     rows = min((bs + 3) // 4, 4)
#     fig = plt.figure(figsize=(columns * 4, rows * 4))
#     for i in range(rows):
#         for j in range(columns):
#             idx = i + j * columns
#             fig.add_subplot(rows, columns, idx + 1)
#             plt.axis('off')
#             plt.imshow((x[idx, :, :, :3] * 255).astype(np.int))
#     plt.show()
#
#
# display_imgs(np.asarray(md.trn_ds.denorm(x)))

# =======================================================================================================================
# Stats
# =======================================================================================================================
#
# x_tot = np.zeros(4)
# x2_tot = np.zeros(4)
# for x,y in iter(md.trn_dl):
#     tmp =  md.trn_ds.denorm(x).reshape(16,-1)
#     x = md.trn_ds.denorm(x).reshape(-1,4)
#     x_tot += x.mean(axis=0)
#     x2_tot += (x**2).mean(axis=0)
#
# channel_avr = x_tot/len(md.trn_dl)
# channel_std = np.sqrt(x2_tot/len(md.trn_dl) - channel_avr**2)
# print(channel_avr,channel_std)

# =======================================================================================================================
# Functions and metrics
# =======================================================================================================================

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


def fbeta_torch(y_true, y_pred, beta=1, threshold=0.5, eps=1e-9):
    y_pred = (y_pred.float() > threshold).float()
    y_true = y_true.float()
    tp = (y_pred * y_true).sum(dim=1)
    precision = tp / (y_pred.sum(dim=1) + eps)
    recall = tp / (y_true.sum(dim=1) + eps)
    return torch.mean(
        precision * recall / (precision * (beta ** 2) + recall + eps) * (1 + beta ** 2))


# =======================================================================================================================

def acc(preds, targs, th=0.5):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds == targs).float().mean()


def recall(preds, targs, thresh=0.5):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    tp = tpos.sum().item()
    tr = targs.sum().item()
    return float(tp + 0.000001) / float(tr + 0.000001)


def precision(preds, targs, thresh=0.5):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    tp = tpos.sum().item()
    pp = pred_pos.sum().item()
    return float(tp + 0.000001) / float(pp + 0.000001)


def fbeta(preds, targs, beta, thresh=0.5):
    """Calculates the F-beta score (the weighted harmonic mean of precision and recall).
    This is the micro averaged version where the true positives, false negatives and
    false positives are calculated globally (as opposed to on a per label basis).

    beta == 1 places equal weight on precision and recall, b < 1 emphasizes precision and
    beta > 1 favors recall.
    """
    assert beta > 0, 'beta needs to be greater than 0'
    beta2 = beta ** 2
    rec = recall(preds, targs, thresh)
    prec = precision(preds, targs, thresh)
    return float((1 + beta2) * prec * rec) / float(beta2 * prec + rec + 0.00000001)


def f1(preds, targs, thresh=0.5): return float(fbeta(preds, targs, 1, thresh))


# =======================================================================================================================
# Training
# =======================================================================================================================

class ConvnetBuilder_custom():
    def __init__(self, f, c, is_multi, is_reg, ps=None, xtra_fc=None, xtra_cut=0,
                 custom_head=None, pretrained=True):
        self.f, self.c, self.is_multi, self.is_reg, self.xtra_cut = f, c, is_multi, is_reg, xtra_cut
        if xtra_fc is None: xtra_fc = [512]
        if ps is None: ps = [0.25] * len(xtra_fc) + [0.5]
        self.ps, self.xtra_fc = ps, xtra_fc

        if f in model_meta:
            cut, self.lr_cut = model_meta[f]
        else:
            cut, self.lr_cut = 0, 0
        cut -= xtra_cut
        layers = cut_model(f(pretrained), cut)

        # replace first convolutional layer by 4->64 while keeping corresponding weights
        # and initializing new weights with zeros
        #w = layers[0].weight
        #layers[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #layers[0].weight = torch.nn.Parameter(torch.cat((w, torch.zeros(64, 1, 7, 7)), dim=1))

        self.nf = model_features[f] if f in model_features else (num_features(layers) * 2)
        if not custom_head: layers += [AdaptiveConcatPool2d(), Flatten()]
        self.top_model = nn.Sequential(*layers)

        n_fc = len(self.xtra_fc) + 1
        if not isinstance(self.ps, list): self.ps = [self.ps] * n_fc

        if custom_head:
            fc_layers = [custom_head]
        else:
            fc_layers = self.get_fc_layers()
        self.n_fc = len(fc_layers)
        self.fc_model = to_gpu(nn.Sequential(*fc_layers))
        if not custom_head: apply_init(self.fc_model, kaiming_normal)
        self.model = to_gpu(nn.Sequential(*(layers + fc_layers)))

    @property
    def name(self):
        return f'{self.f.__name__}_{self.xtra_cut}'

    def create_fc_layer(self, ni, nf, p, actn=None):
        res = [nn.BatchNorm1d(num_features=ni)]
        if p: res.append(nn.Dropout(p=p))
        res.append(nn.Linear(in_features=ni, out_features=nf))
        if actn: res.append(actn)
        return res

    def get_fc_layers(self):
        res = []
        ni = self.nf
        for i, nf in enumerate(self.xtra_fc):
            res += self.create_fc_layer(ni, nf, p=self.ps[i], actn=nn.ReLU())
            ni = nf
        final_actn = nn.Sigmoid() if self.is_multi else nn.LogSoftmax()
        if self.is_reg: final_actn = None
        res += self.create_fc_layer(ni, self.c, p=self.ps[-1], actn=final_actn)
        return res

    def get_layer_groups(self, do_fc=False):
        if do_fc:
            return [self.fc_model]
        idxs = [self.lr_cut]
        c = children(self.top_model)
        if len(c) == 3: c = children(c[0]) + c[1:]
        lgs = list(split_by_idxs(c, idxs))
        return lgs + [self.fc_model]


class ConvLearner(Learner):
    def __init__(self, data, models, precompute=False, **kwargs):
        self.precompute = False
        super().__init__(data, models, **kwargs)
        if hasattr(data, 'is_multi') and not data.is_reg and self.metrics is None:
            self.metrics = [accuracy_thresh(0.5)] if self.data.is_multi else [accuracy]
        if precompute: self.save_fc1()
        self.freeze()
        self.precompute = precompute

    def _get_crit(self, data):
        if not hasattr(data, 'is_multi'): return super()._get_crit(data)

        return F.l1_loss if data.is_reg else F.binary_cross_entropy if data.is_multi else F.nll_loss

    @classmethod
    def pretrained(cls, f, data, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None, precompute=False,
                   pretrained=True, **kwargs):
        models = ConvnetBuilder_custom(f, data.c, data.is_multi, data.is_reg,
                                       ps=ps, xtra_fc=xtra_fc, xtra_cut=xtra_cut, custom_head=custom_head,
                                       pretrained=pretrained)
        return cls(data, models, precompute, **kwargs)

    @classmethod
    def lsuv_learner(cls, f, data, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None, precompute=False,
                     needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=False, **kwargs):
        models = ConvnetBuilder(f, data.c, data.is_multi, data.is_reg,
                                ps=ps, xtra_fc=xtra_fc, xtra_cut=xtra_cut, custom_head=custom_head, pretrained=False)
        convlearn = cls(data, models, precompute, **kwargs)
        convlearn.lsuv_init()
        return convlearn

    @property
    def model(self):
        return self.models.fc_model if self.precompute else self.models.model

    def half(self):
        if self.fp16: return
        self.fp16 = True
        if type(self.model) != FP16: self.models.model = FP16(self.model)
        if not isinstance(self.models.fc_model, FP16): self.models.fc_model = FP16(self.models.fc_model)

    def float(self):
        if not self.fp16: return
        self.fp16 = False
        if type(self.models.model) == FP16: self.models.model = self.model.module.float()
        if type(self.models.fc_model) == FP16: self.models.fc_model = self.models.fc_model.module.float()

    @property
    def data(self):
        return self.fc_data if self.precompute else self.data_

    def create_empty_bcolz(self, n, name):
        return bcolz.carray(np.zeros((0, n), np.float32), chunklen=1, mode='w', rootdir=name)

    def set_data(self, data, precompute=False):
        super().set_data(data)
        if precompute:
            self.unfreeze()
            self.save_fc1()
            self.freeze()
            self.precompute = True
        else:
            self.freeze()

    def get_layer_groups(self):
        return self.models.get_layer_groups(self.precompute)

    def summary(self):
        precompute = self.precompute
        self.precompute = False
        res = super().summary()
        self.precompute = precompute
        return res

    def get_activations(self, force=False):
        tmpl = f'_{self.models.name}_{self.data.sz}.bc'
        # TODO: Somehow check that directory names haven't changed (e.g. added test set)
        names = [os.path.join(self.tmp_path, p + tmpl) for p in ('x_act', 'x_act_val', 'x_act_test')]
        if os.path.exists(names[0]) and not force:
            self.activations = [bcolz.open(p) for p in names]
        else:
            self.activations = [self.create_empty_bcolz(self.models.nf, n) for n in names]

    def save_fc1(self):
        self.get_activations()
        act, val_act, test_act = self.activations
        m = self.models.top_model
        if len(self.activations[0]) != len(self.data.trn_ds):
            predict_to_bcolz(m, self.data.fix_dl, act)
        if len(self.activations[1]) != len(self.data.val_ds):
            predict_to_bcolz(m, self.data.val_dl, val_act)
        if self.data.test_dl and (len(self.activations[2]) != len(self.data.test_ds)):
            if self.data.test_dl: predict_to_bcolz(m, self.data.test_dl, test_act)

        self.fc_data = ImageClassifierData.from_arrays(self.data.path,
                                                       (act, self.data.trn_y), (val_act, self.data.val_y), self.data.bs,
                                                       classes=self.data.classes,
                                                       test=test_act if self.data.test_dl else None, num_workers=8)

    def freeze(self):
        self.freeze_to(-1)

    def unfreeze(self):
        self.freeze_to(0)
        self.precompute = False

    def predict_array(self, arr):
        precompute = self.precompute
        self.precompute = False
        pred = super().predict_array(arr)
        self.precompute = precompute
        return pred


# =======================================================================================================================
# Training params
# =======================================================================================================================

sz = 512  # image size
bs = 24  # batch size
md = get_data(sz, bs)

learner = ConvLearner.pretrained(arch, md, ps=0.4)
learner.opt_fn = optim.Adam
learner.clip = 1.0
learner.crit = FocalLoss()
learner.metrics = [precision, recall, f1]
#print(learner.summary)

# tb_logger = TensorboardLogger(learner.model, md, "res34_512_unbalanced_ext_3chn", metrics_names=["precision", 'recall', 'f1'])
#
# lr = 2e-2
# #learner.f
#
# lrs = np.array([lr / 10, lr / 3, lr])
# =======================================================================================================================
#learner.load('ResNetXt50_512_1')
#learner.unfreeze()

# learner.fit(lrs / 4, 4, cycle_len=2, use_clr=(10, 20), cycle_save_name='res34_512_ext_3chn_1', callbacks=[tb_logger])
# learner.fit(lrs / 4, 2, cycle_len=4, use_clr=(10, 20), cycle_save_name='res34_512_ext_3chn_2', callbacks=[tb_logger])
# learner.fit(lrs / 16, 1, cycle_len=8, use_clr=(5, 20), cycle_save_name='res34_512_ext_3chn_3', callbacks=[tb_logger])


#=======================================================================================================================
# learner.load('res34_512_ext_3chn_3_cyc_0')
# learner.fit(lrs / 16, 1, cycle_len=8, use_clr=(5, 20), cycle_save_name='res34_512_ext_3chn_4', callbacks=[tb_logger])
########################################################################################################################

#learner.load('res34_512_ext_3chn_4_cyc_0')
#learner.fit(lrs / 16, 1, cycle_len=8, use_clr=(5, 20), cycle_save_name='res34_512_ext_3chn_5', use_swa=True, swa_start=1, callbacks=[tb_logger])

#=======================================================================================================================
models = ['res34_512_ext_3chn_4_cyc_0_cp1', 'res34_512_ext_3chn_5_cyc_0-swa_cp2']
#models = ['res34_512_ext_3chn_4_cyc_0_cp1']
########################################################################################################################
# Bagging
########################################################################################################################

for imodel in models:
    print('model loaded:' ,imodel)

    learner.load(imodel)

    # ======================================================================================================================
    # Validation
    # =======================================================================================================================

    def sigmoid_np(x):
        return 1.0 / (1.0 + np.exp(-x))

    print('valid predicting')
    preds, y = learner.TTA(n_aug=1)
    preds = np.stack(preds, axis=-1)
    preds = sigmoid_np(preds)
    pred = preds.max(axis=-1)


    def F1_soft(preds, targs, th=0.5, d=50.0):
        preds = sigmoid_np(d * (preds - th))
        targs = targs.astype(np.float)
        score = 2.0 * (preds * targs).sum(axis=0) / ((preds + targs).sum(axis=0) + 1e-6)
        return score


    def fit_val(x, y):
        params = 0.5 * np.ones(len(name_label_dict))
        wd = 1e-5
        error = lambda p: np.concatenate((F1_soft(x, y, p) - 1.0,
                                          wd * (p - 0.5)), axis=None)
        p, success = opt.leastsq(error, params)
        return p


    th = fit_val(pred, y)
    th[th < 0.1] = 0.1
    print('Thresholds: ', th)
    print('F1 macro: ', f1_score(y, pred > th, average='macro'))
    print('F1 macro (th = 0.5): ', f1_score(y, pred > 0.5, average='macro'))
    print('F1 micro: ', f1_score(y, pred > th, average='micro'))

    print('Fractions: ', (pred > th).mean(axis=0))
    print('Fractions (true): ', (y > th).mean(axis=0))

    # =======================================================================================================================
    # Submission
    # =======================================================================================================================
    print('test prediction')
    SUB_DIR = 'subs/external/' + imodel
    if not os.path.exists(SUB_DIR):
        os.makedirs(SUB_DIR)

    preds_t, y_t = learner.TTA(n_aug=16, is_test=True)
    preds_t = np.stack(preds_t, axis=-1)
    preds_t = sigmoid_np(preds_t)
    pred_t = preds_t.max(axis=-1)  # max works better for F1 macro score

    names = np.array(learner.data.test_ds.fnames)
    save_pkl = np.column_stack((names, pred_t))

    fileName = SUB_DIR + '/' + 'preds.pkl'
    fileObject = open(fileName, 'wb')
    pkl.dump(save_pkl, fileObject)
    fileObject.close()

    with open(fileName, 'rb') as f:
         a = pickle.load(f)

    print('TTA done')

    def save_pred(pred, th=0.5, fname='protein_classification.csv'):
        pred_list = []
        for line in pred:
            s = ' '.join(list([str(i) for i in np.nonzero(line > th)[0]]))
            pred_list.append(s)

        sample_df = pd.read_csv(SAMPLE)
        sample_list = list(sample_df.Id)
        pred_dic = dict((key, value) for (key, value)
                        in zip(learner.data.test_ds.fnames, pred_list))
        pred_list_cor = [pred_dic[id] for id in sample_list]
        df = pd.DataFrame({'Id': sample_list, 'Predicted': pred_list_cor})
        df.to_csv(SUB_DIR + '/' + fname, header=True, index=False)


    # Manual thresholds
    th_t = np.array([0.565, 0.39, 0.55, 0.345, 0.33, 0.39, 0.33, 0.45, 0.38, 0.39,
                     0.34, 0.42, 0.31, 0.38, 0.49, 0.50, 0.38, 0.43, 0.46, 0.40,
                     0.39, 0.505, 0.37, 0.47, 0.41, 0.545, 0.32, 0.1])
    print('Fractions: ', (pred_t > th_t).mean(axis=0))
    save_pred(pred_t, th_t)  # From manual threshold

    # Automatic fitting the thresholds based on the public LB statistics.
    lb_prob = [
        0.362397820, 0.043841336, 0.075268817, 0.059322034, 0.075268817,
        0.075268817, 0.043841336, 0.075268817, 0.010000000, 0.010000000,
        0.010000000, 0.043841336, 0.043841336, 0.014198783, 0.043841336,
        0.010000000, 0.028806584, 0.014198783, 0.028806584, 0.059322034,
        0.010000000, 0.126126126, 0.028806584, 0.075268817, 0.010000000,
        0.222493880, 0.028806584, 0.010000000]


    # I replaced 0 by 0.01 since there may be a rounding error leading to 0

    def Count_soft(preds, th=0.5, d=50.0):
        preds = sigmoid_np(d * (preds - th))
        return preds.mean(axis=0)


    def fit_test(x, y):
        params = 0.5 * np.ones(len(name_label_dict))
        wd = 1e-5
        error = lambda p: np.concatenate((Count_soft(x, p) - y,
                                          wd * (p - 0.5)), axis=None)
        p, success = opt.leastsq(error, params)
        return p


    th_t = fit_test(pred_t, lb_prob)
    th_t[th_t < 0.1] = 0.1
    print('Thresholds: ', th_t)
    print('Fractions: ', (pred_t > th_t).mean(axis=0))
    print('Fractions (th = 0.5): ', (pred_t > 0.5).mean(axis=0))

    save_pred(pred_t, th_t, 'protein_classification_f.csv')  # based on public lb stats
    save_pred(pred_t, th, 'protein_classification_v.csv')  # based on validation
    save_pred(pred_t, 0.5, 'protein_classification_05.csv')  # based on fixed threshold 0.5

    # =======================================================================================================================
    # using the threshold from validation set for classes not present in the public LB:
    class_list = [8, 9, 10, 15, 20, 24, 27]
    for i in class_list:
        th_t[i] = th[i]
    save_pred(pred_t, th_t, 'protein_classification_c.csv')

    # =======================================================================================================================
    # fitting thresholds based on the frequency of classes in the train dataset:
    labels = pd.read_csv(LABELS_org).set_index('Id')
    label_count = np.zeros(len(name_label_dict))
    for label in labels['Target']:
        l = [int(i) for i in label.split()]
        label_count += np.eye(len(name_label_dict))[l].sum(axis=0)
    label_fraction = label_count.astype(np.float) / len(labels)
    print(label_count, label_fraction)

    th_t = fit_test(pred_t, label_fraction)
    th_t[th_t < 0.05] = 0.05
    print('Thresholds: ', th_t)
    print('Fractions: ', (pred_t > th_t).mean(axis=0))
    save_pred(pred_t, th_t, 'protein_classification_t.csv')  # based on frquency of classes in train
    # =======================================================================================================================# res34
# F1 macro:  0.7339006427813839
# F1 macro (th = 0.5):  0.6669998135148151
# F1 micro:  0.7723082957442635

# res101xt-m4
# Thresholds:  [0.54491 0.75237 0.58362 0.55942 0.56169 0.52287 0.56564 0.58306 0.50261 0.52049 0.46712 0.5479  0.57008
#  0.71485 0.59936 0.1     0.66235 0.58874 0.51545 0.51548 0.52326 0.49656 0.65905 0.54701 0.68219 0.50362
#  0.48294 0.29036]
# F1 macro:  0.7011295048856508
# F1 macro (th = 0.5):  0.6415093521306193
# F1 micro:  0.7883417085427137

# # resnet101xt-swa
# [0.52095 0.67501 0.46876 0.62209 0.52894 0.55665 0.55442 0.48154 0.46129 0.75715 0.43572 0.586   0.64507
#  0.64826 0.55982 0.1     0.83022 0.90441 0.55107 0.51155 0.52846 0.4664  0.74345 0.52408 0.79122 0.46872
#  0.55224 0.1    ]
# F1 macro:  0.6819705865476475
# F1 macro (th = 0.5):  0.6223916910357309
# F1 micro:  0.7857503279846604
# sub_0.05 lb: 0.492

########################################################################################################################
# res34 grn+ 512 revised val
# Thresholds:  [0.54664 0.5475  0.55301 0.50708 0.47384 0.5289  0.44358 0.5137  0.41192 0.4685  0.42144 0.54451 0.5089
#  0.53344 0.47533 0.12029 0.39405 0.40774 0.45618 0.46368 0.40192 0.47281 0.56206 0.49217 0.51224 0.49178
#  0.1     0.34105]
# F1 macro:  0.6304911952781457 # 0.451
# F1 macro (th = 0.5):  0.5779850745288859 # 0.438
# F1 micro:  0.6733807952769578


# res34 grn+0.5 swa 4x4 revised val
# Thresholds:  [0.54103 0.56479 0.53996 0.55322 0.47398 0.54292 0.46768 0.53656 0.38274 0.48946 0.41035 0.4907  0.48226
#  0.5141  0.51645 0.1427  0.38583 0.42499 0.45048 0.4634  0.41046 0.45131 0.52466 0.51523 0.67688 0.48726
#  0.45562 0.34339]
# F1 macro:  0.666934899747442
# F1 macro (th = 0.5):  0.5910769947924901
# F1 micro:  0.7727588603196666

# grn34+0.9 swa 4x4 revised val
# Thresholds:  [0.54138 0.56821 0.57645 0.49649 0.45076 0.5454  0.49167 0.52807 0.4007  0.43375 0.37413 0.52472 0.52156
#  0.44734 0.54172 0.1312  0.43421 0.42853 0.46424 0.48458 0.4138  0.45056 0.55984 0.50826 0.71608 0.48222
#  0.51216 0.35996]
# F1 macro:  0.6830337665787448
# F1 macro (th = 0.5):  0.6158043015502873
# F1 micro:  0.7779433681073026

# grn34+0.5 256 revised val
# Thresholds:  [0.53031 0.61858 0.58287 0.50504 0.56897 0.6039  0.48341 0.57169 0.48902 0.61432 0.53577 0.60106 0.52176
#  0.49809 0.59424 0.1445  0.46618 0.50464 0.50754 0.53109 0.51841 0.51343 0.50707 0.58757 0.57555 0.49086
#  0.53558 0.52238]
# F1 macro:  0.6661630089375489
# F1 macro (th = 0.40):  0.5571912452206964
# F1 macro (th = 0.45):  0.6171187769631362
# F1 macro (th = 0.50):  0.6415479086760095 # 0.470
# F1 macro (th = 0.55):  0.6489516914324852
# F1 macro (th = 0.60):  0.6026006073694331
# F1 macro (th = 0.65):  0.553418550910492
# F1 micro:  0.7577577577577577
# Fractions:  [0.4482  0.03153

# wrn - validation-stratified run 1
# Thresholds:  [0.54149 0.55812 0.56723 0.58857 0.55518 0.59766 0.5105  0.59866 0.73868 0.69084 0.56855 0.66444 0.57103
#  0.73118 0.5686  0.63132 0.61924 0.57267 0.54284 0.48375 0.53864 0.47911 0.57397 0.56129 0.78236 0.1
#  0.55899 0.36679]
# F1 macro:  0.7199049787804883
# F1 macro (th = 0.5):  0.6730420723769626  # 0.
# F1 micro:  0.688504734639947

# wrn - validation-stratified run 2 -long -16 more epoch
# Thresholds:  [0.56716 0.63661 0.59092 0.62034 0.52683 0.56752 0.50399 0.61647 0.65823 0.57482 0.52132 0.68148 0.60175
#  0.57967 0.5999  0.61468 0.48772 0.56341 0.5741  0.49707 0.5276  0.49113 0.57197 0.54825 0.62061 0.49563
#  0.624   0.27953]
# F1 macro:  0.7361542316545664
# F1 macro (th = 0.5):  0.6932206090395802
# F1 micro:  0.7887470695493618

# wrn - validation-stratified run 3 -long -balanced train - 16 more epoch --overfit
# 7
# 0.220173
# 0.761697
# 0.856499
# 0.610779
# 0.706326
# Thresholds:  [0.55793 0.78152 0.66333 0.6825  0.65572 0.75289 0.74526 0.63238 0.53035 0.36548 0.42722 0.72039 0.71662
#  0.75241 0.66181 0.40304 0.78582 0.87507 0.72847 0.61643 0.81325 0.5994  0.64994 0.60596 0.8597  0.52862
#  0.84814 0.13508]
# F1 macro:  0.720740962219489
# F1 macro (th = 0.5):  0.6314560071256061
# F1 micro:  0.7730109204368174


# wrn - validation-stratified run 4 -long -unbalanced train - 16 more epoch --overfit Augmentation+
# 7      0.257098   0.746422   0.866529   0.583308   0.688742
# Thresholds:  [0.5979  0.73537 0.634   0.71561 0.69005 0.69933 0.65443 0.63613 0.8177  0.42255 0.4162  0.7414  0.76869
#  0.78629 0.73087 0.45202 0.83717 0.77659 0.72559 0.63488 0.70433 0.60525 0.71827 0.61926 0.76373 0.52041
#  0.78393 0.16647]
# F1 macro:  0.7102068815363216
# F1 macro (th = 0.5):  0.6060142764385398
# F1 micro:  0.
#


# wrn - validation-stratified run 4 -long -unbalanced train - 16 more epoch -grn+
# 7      0.309054   0.610084   0.917235   0.591056   0.710322
# Thresholds:  [0.56554 0.67176 0.63116 0.58562 0.54196 0.63124 0.54626 0.61112 0.60392 0.51268 0.4329  0.70066 0.63997
# 0.71733 0.63157 0.55963 0.64437 0.76405 0.69452 0.5257  0.57838 0.52668 0.55164 0.60304 0.77591 0.50272
# 0.53615 0.41276]
# F1 macro:  0.747663269393283
# F1 macro (th = 0.5):  0.6990677584721688
# F1 micro:  0.7917927134026042

# wrn- orig+extra
#     3      0.639391   0.644622   0.919603   0.429072   0.557231
# Thresholds:  [0.51907 0.62177 0.45659 0.50118 0.48359 0.43588 0.47132 0.45108 0.57427 0.42798 0.47577 0.65654 0.50251
#  0.66157 0.56521 0.30597 0.67259 0.91345 0.51684 0.4059  0.52206 0.44077 0.46492 0.45323 0.66582 0.49714
#  0.5794  0.36993]
# F1 macro:  0.6287293685786259
# F1 macro (th = 0.5):  0.5955509229380472
# F1 micro:  0.7553213304351394

# wrn- orig+extra-semi
# F1 macro:  0.6420030358223189
# F1 macro (th = 0.5):  0.6193720143466512
# F1 micro:  0.7587781548904659