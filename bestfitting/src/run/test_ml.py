# coding: utf-8
import sys
sys.path.insert(0, '..')
import argparse
from tqdm import tqdm

import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torch.nn import DataParallel
from torch.autograd import Variable

from config.config import *
from utils.common_util import *
from networks.imagemlnet import init_network
from datasets.protein_ml_dataset import ProteinMLDataset
from utils.augment_util import *
from utils.log_util import Logger

datasets_names = ['train', 'val', 'test', 'ext']

parser = argparse.ArgumentParser(description='PyTorch Protein Learning')
parser.add_argument('--out_dir', type=str, help='destination where predicted result should be saved')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id used for predicting (default: 0)')
parser.add_argument('--arch', default='class_densenet121_dropout', type=str,
                    help='model architecture (default: class_densenet121_dropout)')
parser.add_argument('--num_classes', default=12815, type=int, help='number of classes (default: 28)')
parser.add_argument('--in_channels', default=4, type=int, help='in channels (default: 4)')
parser.add_argument('--img_size', default=768, type=int, help='image size (default: 768)')
parser.add_argument('--batch_size', default=32, type=int, help='train mini-batch size (default: 32)')
parser.add_argument('--workers', default=3, type=int, help='number of data loading workers (default: 3)')
parser.add_argument('--dataset', default='test', type=str, choices=datasets_names,
                    help='predict dataset: ' + ' | '.join(datasets_names) + ' (default: test)')
parser.add_argument('--predict_epoch', default=None, type=int, help='number epoch to predict')

def main():
    args = parser.parse_args()

    log_out_dir = opj(RESULT_DIR, 'logs', args.out_dir)
    if not ope(log_out_dir):
        os.makedirs(log_out_dir)
    log = Logger()
    log.open(opj(log_out_dir, 'log.submit.txt'), mode='a')

    args.predict_epoch = 'final' if args.predict_epoch is None else '%03d' % args.predict_epoch
    network_path = opj(RESULT_DIR, 'models', args.out_dir, '%s.pth' % args.predict_epoch)

    submit_out_dir = opj(RESULT_DIR, 'submissions', args.out_dir, 'epoch_%s' % args.predict_epoch)
    log.write(">> Creating directory if it does not exist:\n>> '{}'\n".format(submit_out_dir))
    if not ope(submit_out_dir):
        os.makedirs(submit_out_dir)

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    model_params = {}
    model_params['architecture'] = args.arch
    model_params['num_classes'] = args.num_classes
    model_params['in_channels'] = args.in_channels
    model = init_network(model_params)
    model.set_configs(extract_feature=True)

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
    elif dataset == 'ext':
        test_split_file = opj(DATA_DIR, 'split', 'external_antibody_split.csv')
    elif dataset == 'train':
        test_split_file = opj(DATA_DIR, 'split', 'external_trainset_antibody_split.csv')
    elif dataset == 'val':
        test_split_file = opj(DATA_DIR, 'split', 'external_validset_antibody_split.csv')
    else:
        raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))
    test_dataset = ProteinMLDataset(
        test_split_file,
        img_size=args.img_size,
        is_trainset=False,
        return_label=False,
        in_channels=args.in_channels,
        transform=None,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    with torch.no_grad():
        predict(test_loader, model, submit_out_dir, dataset)

def predict(test_loader, model, submit_out_dir, dataset):
    all_feats = []
    img_ids = np.array(test_loader.dataset.img_ids)
    for it, iter_data in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
        images, indices = iter_data
        images = Variable(images.cuda(), volatile=True)
        outputs = model(images)
        logits, feats = outputs

        all_feats += feats.data.cpu().numpy().tolist()
    img_ids = img_ids[:len(all_feats)]
    all_feats = np.array(all_feats).reshape(len(img_ids), -1)

    np.savez_compressed(opj(submit_out_dir, 'extract_feats_%s.npz' % dataset), feats=all_feats, ids=img_ids)

if __name__ == '__main__':
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    main()
    print('\nsuccess!')
