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
import pickle as pkl

import warnings
warnings.filterwarnings("ignore")

#=======================================================================================================================
# Something
#=======================================================================================================================

PATH = './'
#TRAIN = '../input/HPAv18/png/'
TRAIN = '../input/train/'
TEST = '../input/test/'
LABELS_ext = '../input/HPAv18/HPAv18RBGY_wodpl.csv'
LABELS = '../input/train_org_ext.csv'
#LABELS = '../input/train.csv'
SAMPLE = '../input/sample_submission.csv'

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

nw = 4   #number of workers for data loader
arch = wrn #specify target architecture

#=======================================================================================================================
# Data
#=======================================================================================================================
#TRAIN_IMAGES_PER_CATEGORY = 1000
image_df = pd.read_csv(LABELS)
# remove broken images
image_df = image_df[(image_df.Id != 'dc756dea-bbb4-11e8-b2ba-ac1f6b6435d0') &
                    (image_df.Id != 'c861eb54-bb9f-11e8-b2b9-ac1f6b6435d0') &
                    (image_df.Id != '7a88f200-bbc3-11e8-b2bc-ac1f6b6435d0') &
                    (image_df.Id != '27751_219_G10_1')
]

image_df['target_list'] = image_df['Target'].map(lambda x: [int(a) for a in x.split(' ')])

all_labels = list(chain.from_iterable(image_df['target_list'].values))
c_val = Counter(all_labels)
n_keys = c_val.keys()
max_idx = max(n_keys)

#==================================================================================
# visualize train distribution
# fig, ax1 = plt.subplots(1,1, figsize = (10, 5))
# ax1.bar(n_keys, [c_val[k] for k in n_keys])
# ax1.set_xticks(range(max_idx))
# ax1.set_xticklabels([name_label_dict[k] for k in range(max_idx)], rotation=90)
# plt.show()
#==================================================================================
for k,v in c_val.items():
    print(name_label_dict[k], 'count:', v)

# create a categorical vector
image_df['target_vec'] = image_df['target_list'].map(lambda ck: [i in ck for i in range(max_idx+1)])

raw_train_df, valid_df = train_test_split(image_df,
                 test_size = 0.10,
                  # hack to make stratification work
                 stratify = image_df['Target'].map(lambda x: x[:3] if '27' not in x else '0'),
                                          random_state= 42)

print(raw_train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')

tr_n = raw_train_df['Id'].values.tolist()
val_n = valid_df['Id'].values.tolist()
tr_n = tr_n[:-1]  # pytorch has problems if last batch has one sample

test_names = list({f[:36] for f in os.listdir(TEST)})
# #=================================================================================
# # # Balance data
# #================================================================================
# # keep labels with more then 50 objects
# out_df_list = []
# for k,v in c_val.items():
#     if v>50:
#         keep_rows = raw_train_df['target_list'].map(lambda x: k in x)
#         out_df_list += [raw_train_df[keep_rows].sample(TRAIN_IMAGES_PER_CATEGORY,
#                                                        replace=True)]
# train_df = pd.concat(out_df_list, ignore_index=True)
#
# tr_n = train_df['Id'].values.tolist()
# val_n = valid_df['Id'].values.tolist()
# tr_n = tr_n[:-2]  # pytorch has problems if last batch has one sample
#
# print(train_df.shape[0])
# print(len(tr_n))
# print('unique train:',len(train_df['Id'].unique().tolist()))
#
# #=========================================================================
# #show class graph
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
# train_sum_vec = np.sum(np.stack(raw_train_df['target_vec'].values, 0), 0)
# valid_sum_vec = np.sum(np.stack(valid_df['target_vec'].values, 0), 0)
# ax1.bar(n_keys, [train_sum_vec[k] for k in n_keys])
# ax1.set_title('Training Distribution')
# ax2.bar(n_keys, [valid_sum_vec[k] for k in n_keys])
# ax2.set_title('Validation Distribution')
# plt.show()
#=======================================================================================================================
#=======================================================================================================================
# # old loading
# train_names = list({f[:36] for f in os.listdir(TRAIN)})
# train_names.remove('dc756dea-bbb4-11e8-b2ba-ac1f6b6435d0')
# train_names.sort()
# test_names = list({f[:36] for f in os.listdir(TEST)})
# tr_n, val_n = train_test_split(train_names, test_size=0.1, random_state=42)
#=======================================================================================================================

def open_rgby(path,id): #a function that reads RGBY image
    #print(id)
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    try:
        img = [cv2.imread(os.path.join(path, id+'_'+color+'.png'), flags).astype(np.float32)/255
               for color in colors]
        img = np.stack(img, axis=-1)
        return img
    except:
        print('img broken:', id)

class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.labels = pd.read_csv(LABELS).set_index('Id')
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

def get_data(sz,bs):
    #data augmentation
    aug_tfms = [RandomRotate(30, tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO),
                #RandomCrop(480, tfm_y=TfmType.NO),
                #RandomRotateZoom(deg=30, zoom=1, stretch=0.5)
                ]
    #mean and std in of each channel in the train set
    #Original set: [0.0869  0.0599  0.06534 0.08923] [0.13047 0.09831 0.14875 0.1333 ]
    # Extra data:[0.04445 0.0499  0.01867 0.10086] [0.05364 0.07222 0.02486 0.13921]
    # combined: [0.05697 0.05282 0.03241 0.09748] [0.08622 0.08092 0.08611 0.13758]
    stats = A([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])
    tfms = tfms_from_stats(stats, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
                aug_tfms=aug_tfms)
    ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],TRAIN),
                (val_n,TRAIN), tfms, test=(test_names,TEST))
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    return md

#======================================================================================================================
# Display
#=======================================================================================================================

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

#=======================================================================================================================
# Stats
#=======================================================================================================================
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

#=======================================================================================================================
# Functions and metrics
#=======================================================================================================================

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
    precision = tp / (y_pred.sum(dim=1)+eps)
    recall = tp / (y_true.sum(dim=1)+eps)
    return torch.mean(
        precision*recall / (precision*(beta**2)+recall+eps) * (1+beta**2))

#=======================================================================================================================

def acc(preds,targs,th=0.5):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()

def recall(preds, targs, thresh=0.5):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    tp = tpos.sum().item()
    tr = targs.sum().item()
    return float(tp+0.000001)/float( tr + 0.000001)

def precision(preds, targs, thresh=0.5):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    tp = tpos.sum().item()
    pp = pred_pos.sum().item()
    return float(tp+0.000001)/float(pp + 0.000001)

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
    return float((1 + beta2) * prec * rec) / float(beta2 * prec + rec  + 0.00000001)

def f1(preds, targs, thresh=0.5): return float(fbeta(preds, targs, 1, thresh))

#=======================================================================================================================
# Training
#=======================================================================================================================

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
        w = layers[0].weight
        layers[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        layers[0].weight = torch.nn.Parameter(torch.cat((w, torch.zeros(64, 1, 7, 7)), dim=1))

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

#=======================================================================================================================
# Training params
#=======================================================================================================================
# Pretrain on 256
#=======================================================================================================================
# sz = 256 #image size
# bs = 16  #batch size
# md = get_data(sz,bs)
#
# learner = ConvLearner.pretrained(arch, md, ps=0.2) #dropout 50%
# learner.opt_fn = optim.Adam
# learner.clip = 1.0 #gradient clipping
# learner.crit = FocalLoss()
# learner.metrics = [precision, recall, f1]
# print(learner.summary)

#learner.lr_find()
#learner.sched.plot()
#plt.show()

# tb_logger = TensorboardLogger(learner.model, md, "wrn_256_ext_long", metrics_names=["precision", 'recall', 'f1'])
# lr = 1e-3
# lrs=np.array([lr/10,lr/3,lr])
# #=======================================================================================================================
# learner.load('wrn_512_unbalanced_ext_semi')
# learner.unfreeze()
# learner.fit(lrs/4,4,cycle_len=8,use_clr=(10,20), best_save_name='wrn_256_ext_1', callbacks=[tb_logger])
# learner.fit(lrs/16,4,cycle_len=8,use_clr=(5,20), best_save_name='wrn_256_ext_2', callbacks=[tb_logger])
# learner.save('wrn_256_unbalanced_ext_long')
# #=======================================================================================================================
# # Finetune on 512
# # ======================================================================================================================
sz = 512 #image size
bs = 10  #batch size
md = get_data(sz,bs)

learner = ConvLearner.pretrained(arch, md, ps=0.2)
learner.opt_fn = optim.Adam
learner.clip = 1.0
learner.crit = FocalLoss()
learner.metrics = [precision, recall, f1]
#print(learner.summary)

# tb_logger = TensorboardLogger(learner.model, md, "wrn_512_ext_long", metrics_names=["precision", 'recall', 'f1'])
# lr = 1e-3
# lrs=np.array([lr/10,lr/3,lr])
# #=======================================================================================================================
# learner.load('wrn_512_ext_1')
# learner.unfreeze()
# learner.fit(lrs/4,2,cycle_len=6,use_clr=(10,20), cycle_save_name='wrn_512_ext_1', callbacks=[tb_logger])
# learner.fit(lrs/16,1,cycle_len=4,use_clr=(5,20), cycle_save_name='wrn_512_ext_2', callbacks=[tb_logger])
# learner.save('wrn_512_unbalanced_ext_semi')


# =======================================================================================================================
# Submission
# =======================================================================================================================
models = ['wrn_512_unbalanced_ext_semi']
#=======================================================================================================================
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

    # with open(fileName, 'rb') as f:
    #      a = pickle.load(f)

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
    labels = pd.read_csv(LABELS).set_index('Id')
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
# =======================================================================================================================