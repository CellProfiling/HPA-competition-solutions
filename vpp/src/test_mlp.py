#coding=utf-8

import os
from torch import nn
import warnings
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from utils.feature_utils import read_all_features
import random
import torch
from torch.utils.data import DataLoader

import argparse

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Finetune models Test')

parser.add_argument('--gpuid', '-g', default='0', metavar='N',
                    help='gpu id used for training (default: 0)')
parser.add_argument('--tag', '-t', default='', help='tag used for save')




class BaseConfig():
    def __init__(self,tag=''):
        self.features_type='inceptionv3_inceptionv4_xception_800_offi_hpa'
        self.num_classes = 28
        self.batch_size = 512
        self.sample_submit_file = "./data/sample_submission.csv"
        base_dir= './models/MLP/'
        self.test_file = './data/sample_submission_featureindex_label.csv'
        self.features_test = './features/inceptionv3_inceptionv4_xception_800_10fold/test_offi_hpa_features.h5'
        self.feature_dim = 3840
        print('[features_test]:', self.features_test)
        if tag != '':
            base_dir = base_dir+'/'+tag
        self.weights = base_dir + "/models"
        self.submit = base_dir + "/submit/"
        self.logs_dir = base_dir + '/logs'
        # self.best_models = base_dir+'/bestmodel'
        self.submit_file = os.path.join(self.submit, '{}_submission.csv'.format(self.features_type))

        self.run_type = 'train'
        mkdir_with_check(base_dir)
        mkdir_with_check(self.weights)
        mkdir_with_check(self.submit)
        mkdir_with_check(self.logs_dir)
        # mkdir_with_check(self.best_models)

def mkdir_with_check(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        try:
            os.system('chmod 777 {}'.format(dir_path))
        except:
            pass



class HumanDataset_MLP(Dataset):
    def __init__(self, images_df, mode="test", config=''):
        self.config = config
        self.images_df = images_df.copy()
        self.mlb = MultiLabelBinarizer(classes=np.arange(0, self.config.num_classes))
        self.mlb.fit(np.arange(0, self.config.num_classes))
        self.mode = mode
        if mode=='train':
            self.features = read_all_features(self.config.features_train)
        else:
            self.features = read_all_features(self.config.features_test)


    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):
        if not isinstance(index, int):
            index = index.item()
        feature_index = int(self.images_df.iloc[index].Id)
        X = self.features[feature_index]
        if not self.mode == "test":
            labels = np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))
            y = np.eye(self.config.num_classes, dtype=np.float)[labels].sum(axis=0)
        else:
            y = str(self.images_df.iloc[index].Id)
        X = torch.from_numpy(X)
        return X.float(), y


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(config.feature_dim),
            nn.Dropout(0.5),
            nn.Linear(config.feature_dim, 128))
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, config.num_classes),
        )


    def forward(self, input):

        x_feature = self.fc1(input)
        x_class = self.fc2(x_feature)

        return x_class, x_feature
# 3. test model on public dataset and save the probability matrix
def test(test_loader, model, fold,write_result=True):
    sample_submission_df = pd.read_csv(config.sample_submit_file)
    filenames, labels, submissions = [], [], []
    model.cuda()
    model.eval()
    submit_results = []
    result_score = np.zeros((len(test_loader), config.num_classes))
    for i, (input, filepath) in enumerate(test_loader):
        # 3.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.cuda(non_blocking=True)
            y_pred, sub_features = model(image_var)
            label = y_pred.sigmoid().cpu().data.numpy()
            result_score[i] = label[0]
            # print(label > 0.5)
            labels.append(label > 0.15)
            filenames.append(filepath)

    if write_result:
        for row in np.concatenate(labels):
            subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
            submissions.append(subrow)
        sample_submission_df['Predicted'] = submissions
        sample_submission_df.to_csv(config.submit_file[:-4] + '_fold{}.csv'.format(fold), index=None)
        np.save(config.submit_file[:-4] + '_fold{}_score.npy'.format(fold), result_score)
        print('result has saved to ' + config.submit_file[:-4] + '_fold{}.csv'.format(fold))

# 4. main function
def predict(fold):

    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)

    model = MLP(config)
    model.cuda()

    test_files = pd.read_csv(config.test_file)
    test_gen = HumanDataset_MLP(test_files, mode="test", config=config)
    test_loader = DataLoader(test_gen, 1, shuffle=False, pin_memory=True, num_workers=0)
    model_path=os.path.join(config.weights,'{}_fold_{}_model_best_loss.pth.tar'.format(config.features_type,fold))

    print('load model from:'+model_path)

    best_model = torch.load(model_path)
    model.load_state_dict(best_model["state_dict"])
    test(test_loader, model, fold)

def main():
    for i in xrange(10):
        predict(i)

if __name__ == "__main__":
    args = parser.parse_args()
    gpu_id = args.gpuid
    config = BaseConfig(tag=args.tag)
    # model_input=args.model_path
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    main()
