# add oversampling
# new image format

import glob
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

import pandas as pd

from HPAdataset_01 import HPADataset
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils import data
from utils import FocalLoss, AverageMeter, np_macro_f1, get_learning_rate, save_checkpoint, load_checkpoint
from utils import Logger
from sklearn.metrics import f1_score
from torch.nn import functional as F

from torchvision import transforms

# Some parameters
Nfold = 0
epoch = 40
debug = False
#first = 50
load = False
learning_rate = 0.001
if load:
    learning_rate = 0.005
if debug:
    bat_size = 16
    img_size = 128
else:
    bat_size = 96
    img_size = 512
model_name = 'inceptionv3'
if model_name=='seresnext50':
    from models.seresnext50_ds import Model
if model_name =='inceptionv3':
    from models.inceptionv3_ds import Model


directory = '../data/'
train_path = os.path.join(directory, 'processed_train')
model = Model()
if not debug:
    model = nn.DataParallel(model)  # modified by wyh

test_prediction_manyfold = []
test_path = os.path.join(directory, 'processed_test')
sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['ext']=0
test_dataset = HPADataset(test_path, sample_submission, img_size, mode='test',color='bgry')

all_train_prediction = []
all_train_df = []
for Nfold in range(5):
    filename = 'final_{}_ds_fold_{}'.format(model_name,str(Nfold))

    directory = '../data/'
    save_path = '../checkpoint'
    checkpoint = '{}_lr{}_maxEpoch{}.pth'.format(filename, learning_rate, epoch)

    log = Logger()
    log.open('../logs/log-{}'.format(filename),'a+')

    load_checkpoint('../checkpoint/'+checkpoint, model)

    model.cuda()

    # Predict
    val_images = pd.read_csv('../data/fold_%d.csv' % Nfold)
    all_train_df.append(val_images[['Id','Target']])
    val_images['ext'] = val_images['group'].apply(lambda x: 1 if x == 'hpa_rgby' else 0)
    dataset_val = HPADataset(train_path, val_images, img_size, mode='valid', color='rgby',
                             ext_path='../ext_data/moredata_compressed')
    val_loader = data.DataLoader(dataset_val, batch_size=bat_size, shuffle=False, num_workers=4)

    model.eval()

    tta_args = ['null','hflip','vflip','bflip','rotate','gamma']
    test_predictions = []
    val_predictions = []
    for tta in tta_args:
        print('augmentation: %s'%tta)
        dataset_val.set_tta(tta)
        test_dataset.set_tta(tta)

        tta_test_predictions = []
        tta_val_predictions = []
        with torch.no_grad():
            for image in tqdm(data.DataLoader(test_dataset, batch_size=bat_size, num_workers=4)):
                image = image.type(torch.FloatTensor).cuda()
                predict,_,_,_ = model(image)
                predict = F.sigmoid(predict).cpu().data.numpy()
                tta_test_predictions.append(predict)
        tta_test_predictions = np.vstack(tta_test_predictions)
        test_predictions.append(tta_test_predictions)


        # val_predictions = []
        with torch.no_grad():
            for image, _ in val_loader:
                image = image.type(torch.FloatTensor).cuda()
                predict,_,_,_ = model(image)
                predict = F.sigmoid(predict).cpu().numpy()
                tta_val_predictions.append(predict)
        tta_val_predictions = np.vstack(tta_val_predictions)
        val_predictions.append(tta_val_predictions)

    test_prediction_manyfold.append(np.mean(test_predictions,0))
    all_train_prediction.append(np.mean(val_predictions,0))
    # val_predictions = np.concatenate(val_predictions)
    # all_train_prediction.append(val_predictions)

# validation set result
all_train_prediction = np.concatenate(all_train_prediction)
all_train_df = pd.concat(all_train_df).reset_index(drop=True)
np.save('../test_result/{}_oof_result_tta.npy'.format(model_name),all_train_prediction)
# all_train_df.to_csv('../test_result/{}_oof_result.csv'.format(model_name),index=False)

# test set result
test_prediction_manyfold = np.mean(test_prediction_manyfold, axis=0)
threshold = 0.15
labels = (test_prediction_manyfold > threshold).astype(int)
submissions = []
for row in labels:
    subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
    submissions.append(subrow)

    #Save
os.makedirs('../submissions',exist_ok=True)
os.makedirs('../test_result',exist_ok=True)
os.makedirs('../logs',exist_ok=True)
np.save('../test_result/{}_test_5fold_tta'.format(model_name), test_prediction_manyfold)
sample_submission['Predicted'][:len(submissions)] = submissions
sample_submission[['Id','Predicted']].to_csv('../submissions/{}_tta.csv'.format(filename),index=False)
