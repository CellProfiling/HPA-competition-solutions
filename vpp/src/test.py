#coding=utf-8

import os
import random
import warnings
import pandas as pd
import numpy as np

#from utils import *
from reader import HumanDataset
from tqdm import tqdm
from model import *
from config import get_config
import torch
from torch.utils.data import DataLoader
from vpp_solution.utils.feature_utils import write_feature_2_h5

import argparse

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Finetune models Test')
parser.add_argument('--model_type', '-m', help='type of model')

parser.add_argument('--gpuid', '-g', default='0', metavar='N',
                    help='gpu id used for training (default: 0)')
parser.add_argument('--tag', '-t', default='', help='tag used for save')
parser.add_argument('--model_path', '-p', default='',help='path of model')
parser.add_argument('--submit_file_name', '-s', default='',help='name of the submit file')
parser.add_argument('--image_size','-imsize', default=512, type=int,help='image width and height')
parser.add_argument('--batch_size','-bs', default=1, type=int,help='batch size')
parser.add_argument('--num_workers','-work', default=5, type=int,help='number of thread')
parser.add_argument('--with_features', action='store_true',help='if extract features')
parser.add_argument('--cropindex', '-ci', default=-1, type=int, help='crop index')

# 3. test model on public dataset and save the probability matrix
def test(test_loader,model,folds):
    sample_submission_df = pd.read_csv(config.sample_submit_file)
    labels ,submissions= [],[]
    model.cuda()
    model.eval()
    result_score = np.zeros((len(sample_submission_df.Id),config.num_classes))
    if args.with_features:features = np.zeros((len(sample_submission_df.Id),128))
    start=0
    for i,(input,filepath) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            image_var = input.cuda(non_blocking=True)
            if args.with_features:
                y_pred, sub_features = model(image_var)
                sub_features = sub_features.cpu().data.numpy()
                features[start:(start+sub_features.shape[0])]=sub_features
            else:
                y_pred = model(image_var)
            label = y_pred.sigmoid().cpu().data.numpy()
            result_score[start:(start+label.shape[0])]=label
            start=start+label.shape[0]
            labels.extend(label > 0.15)

    for row in labels:
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    print len(submissions)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv(config.submit_file, index=None)
    np.save(config.submit_file[:-4]+'_score.npy',result_score)
    print('result has saved to '+config.submit_file)
    if args.with_features:
        feature_h5_save_path = config.submit_file[:-4] + '_feature.h5'
        if os.path.exists(feature_h5_save_path):
            print('h5 file --> delete')
            os.system('rm -rf  {}'.format(feature_h5_save_path))
        write_feature_2_h5(feature_h5_save_path, features)
# 4. main function
def main():

    fold = 0
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)

    model = get_net(config)
    model.cuda()

    test_files = pd.read_csv(config.sample_submit_file)

    test_gen = HumanDataset(test_files,config.test_data,augument=False,mode="test",config=config)
    test_loader = DataLoader(test_gen,args.batch_size,shuffle=False,pin_memory=True,num_workers=args.num_workers)
    if model_input=='':
        model_path = "%s/%s_fold_%s_model_best_loss.pth.tar" % (config.best_models, config.model_name, str(fold))
    else:
        model_path = model_input
        if not os.path.exists(model_path):
            model_path=os.path.join(config.weights,model_path)

    print('load model from:'+model_path)

    best_model = torch.load(model_path)
    model.load_state_dict(best_model["state_dict"])
    test(test_loader,model,fold)

if __name__ == "__main__":
    args = parser.parse_args()
    gpu_id = args.gpuid
    config = get_config(args.model_type,gpu_id,args.tag)
    config.with_hpa = False
    config.img_weight = args.image_size
    config.img_height = args.image_size
    config.cropindex = args.cropindex
    if args.with_features:config.run_type='feature'
    model_input=args.model_path
    config.set_path('official', config.img_height)
    if args.submit_file_name!='':
        config.submit_file=os.path.join(os.path.dirname(config.submit_file),args.submit_file_name)
    elif model_input!='':
        tmp_fold_epoch=args.model_path.split('flod_')[1].split('.pth')[0].split('_')
        submit_name = '{}_{}_{}_epoch{}.csv'.format(config.model_name,args.tag,tmp_fold_epoch[0],tmp_fold_epoch[-1])
        config.submit_file = os.path.join(os.path.dirname(config.submit_file), submit_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(config.best_models)
    main()
