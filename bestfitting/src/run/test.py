# coding: utf-8
import sys
sys.path.insert(0, '..')
import argparse
from tqdm import tqdm
import pandas as pd

import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.autograd import Variable

from config.config import *
from utils.common_util import *
from networks.imageclsnet import init_network
from datasets.protein_dataset import ProteinDataset
from utils.augment_util import *
from utils.log_util import Logger

datasets_names = ['test', 'val']
split_names = ['random_ext_folds5', 'random_ext_noleak_clean_folds5']
augment_list = ['default', 'flipud', 'fliplr','transpose', 'flipud_lr',
                'flipud_transpose', 'fliplr_transpose', 'flipud_lr_transpose']

parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--out_dir', type=str, help='destination where predicted result should be saved')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id used for predicting (default: 0)')
parser.add_argument('--arch', default='class_densenet121_dropout', type=str,
                    help='model architecture (default: class_densenet121_dropout)')
parser.add_argument('--num_classes', default=28, type=int, help='number of classes (default: 28)')
parser.add_argument('--in_channels', default=4, type=int, help='in channels (default: 4)')
parser.add_argument('--img_size', default=768, type=int, help='image size (default: 768)')
parser.add_argument('--crop_size', default=512, type=int, help='crop size (default: 512)')
parser.add_argument('--batch_size', default=32, type=int, help='train mini-batch size (default: 32)')
parser.add_argument('--workers', default=3, type=int, help='number of data loading workers (default: 3)')
parser.add_argument('--fold', default=0, type=int, help='index of fold (default: 0)')
parser.add_argument('--augment', default='default', type=str, help='test augmentation (default: default)')
parser.add_argument('--seed', default=100, type=int, help='random seed (default: 100)')
parser.add_argument('--seeds', default=None, type=str, help='predict seed')
parser.add_argument('--dataset', default='test', type=str, choices=datasets_names,
                    help='dataset options: ' + ' | '.join(datasets_names) + ' (default: test)')
parser.add_argument('--split_name', default='random_ext_folds5', type=str, choices=split_names,
                    help='split name options: ' + ' | '.join(split_names) + ' (default: random_ext_folds5)')
parser.add_argument('--predict_epoch', default=None, type=int, help='number epoch to predict')

def main():
    args = parser.parse_args()

    log_out_dir = opj(RESULT_DIR, 'logs', args.out_dir, 'fold%d' % args.fold)
    if not ope(log_out_dir):
        os.makedirs(log_out_dir)
    log = Logger()
    log.open(opj(log_out_dir, 'log.submit.txt'), mode='a')

    args.predict_epoch = 'final' if args.predict_epoch is None else '%03d' % args.predict_epoch
    network_path = opj(RESULT_DIR, 'models', args.out_dir, 'fold%d' % args.fold, '%s.pth' % args.predict_epoch)

    submit_out_dir = opj(RESULT_DIR, 'submissions', args.out_dir, 'fold%d' % args.fold, 'epoch_%s' % args.predict_epoch)
    log.write(">> Creating directory if it does not exist:\n>> '{}'\n".format(submit_out_dir))
    if not ope(submit_out_dir):
        os.makedirs(submit_out_dir)

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    args.augment = args.augment.split(',')
    for augment in args.augment:
        if augment not in augment_list:
            raise ValueError('Unsupported or unknown test augmentation: {}!'.format(augment))

    model_params = {}
    model_params['architecture'] = args.arch
    model_params['num_classes'] = args.num_classes
    model_params['in_channels'] = args.in_channels
    model = init_network(model_params)

    log.write(">> Loading network:\n>>>> '{}'\n".format(network_path))
    checkpoint = torch.load(network_path)
    model.load_state_dict(checkpoint['state_dict'])
    log.write(">>>> loaded network:\n>>>> epoch {}\n".format(checkpoint['epoch']))

    # moving network to gpu and eval mode
    model = DataParallel(model)
    model.cuda()
    model.eval()

    # Data loading code
    dataset = args.dataset
    if dataset == 'test':
        test_split_file = opj(DATA_DIR, 'split', 'test_11702.csv')
    elif dataset == 'val':
        test_split_file = opj(DATA_DIR, 'split', args.split_name, 'random_valid_cv%d.csv' % args.fold)
    else:
        raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))
    test_dataset = ProteinDataset(
        test_split_file,
        img_size=args.img_size,
        is_trainset=(dataset != 'test'),
        return_label=False,
        in_channels=args.in_channels,
        transform=None,
        crop_size=args.crop_size,
        random_crop=False,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    seeds = [args.seed] if args.seeds is None else [int(i) for i in args.seeds.split(',')]
    for seed in seeds:
        test_dataset.random_crop = (seed != 0)
        for augment in args.augment:
            test_loader.dataset.transform = eval('augment_%s' % augment)
            if args.crop_size > 0:
                sub_submit_out_dir = opj(submit_out_dir, '%s_seed%d' % (augment, seed))
            else:
                sub_submit_out_dir = opj(submit_out_dir, augment)
            if not ope(sub_submit_out_dir):
                os.makedirs(sub_submit_out_dir)
            with torch.no_grad():
                predict(test_loader, model, sub_submit_out_dir, dataset)

def predict(test_loader, model, submit_out_dir, dataset):
    all_probs = []
    img_ids = np.array(test_loader.dataset.img_ids)
    for it, iter_data in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
        images, indices = iter_data
        images = Variable(images.cuda(), volatile=True)
        outputs = model(images)
        logits = outputs

        probs = F.sigmoid(logits).data
        all_probs += probs.cpu().numpy().tolist()
    img_ids = img_ids[:len(all_probs)]
    all_probs = np.array(all_probs).reshape(len(img_ids), -1)

    np.save(opj(submit_out_dir, 'prob_%s.npy' % dataset), all_probs)

    result_df = prob_to_result(all_probs, img_ids)
    result_df.to_csv(opj(submit_out_dir, 'results_%s.csv.gz' % dataset), index=False, compression='gzip')

def prob_to_result(probs, img_ids, th=0.5):
    probs = probs.copy()
    probs[np.arange(len(probs)), np.argmax(probs, axis=1)] = 1

    pred_list = []
    for line in probs:
        s = ' '.join(list([str(i) for i in np.nonzero(line > th)[0]]))
        pred_list.append(s)
    result_df = pd.DataFrame({ID: img_ids, PREDICTED: pred_list})
    return result_df

if __name__ == '__main__':
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    main()
    print('\nsuccess!')
