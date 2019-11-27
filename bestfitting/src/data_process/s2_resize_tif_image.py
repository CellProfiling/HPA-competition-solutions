import sys
sys.path.insert(0, '..')
import cv2
import mlcrate as mlc
import argparse

from config.config import *
from utils.common_util import *
from PIL import Image

def do_convert(fname_tif):
    img = np.array(Image.open(opj(tif_dir, fname_tif)))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    fname_img = fname_tif.replace('.tif', '.png')
    cv2.imwrite(opj(img_dir, fname_img), img)

parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--dataset', type=str, default='train', help='dataset')
parser.add_argument('--size', type=int, default=1536, help='size')
args = parser.parse_args()

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    size = args.size
    dataset = args.dataset
    tif_dir = opj(TIF_DIR, dataset, 'tifs')
    img_dir = opj(DATA_DIR, dataset, 'images_%d' % size)
    n_cpu = 3

    os.makedirs(img_dir, exist_ok=True)
    start_num = max(0, len(os.listdir(img_dir)) - n_cpu * 2)
    fnames = np.sort(os.listdir(tif_dir))[start_num:]
    pool = mlc.SuperPool(n_cpu)
    df_list = pool.map(do_convert, fnames, description='resize %s image' % dataset)

    print('\nsuccess!')
