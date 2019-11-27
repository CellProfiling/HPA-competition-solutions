import sys
sys.path.insert(0, '..')
import cv2
import mlcrate as mlc
import argparse
from PIL import Image

from config.config import *
from utils.common_util import *

parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--size', type=int, default=1536, help='size')
args = parser.parse_args()

def do_resize(param):
    src, fname, dst, size  = param
    color = fname.replace('.jpg','').split('_')[-1]
    try:
        im = np.array(Image.open(opj(src, fname)))[:, :, COLOR_INDEXS.get(color)]
    except:
        print('bad image : %s' % fname)
        im = cv2.imread(opj(src, fname))[:, :, -1::-1][:, :, COLOR_INDEXS.get(color)]
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(opj(dst, fname), im,  [int(cv2.IMWRITE_JPEG_QUALITY), 85])

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    size = args.size
    src = opj(EXTERNEL_DIR, 'HPAv18/jpg')
    dst = opj(DATA_DIR, 'train/external_v18_%d' % size)

    fnames = np.sort(os.listdir(src))
    os.makedirs(dst, exist_ok=True)
    start_num = max(0, len(os.listdir(dst)) - 48)
    fnames = fnames[start_num:]
    params = [(src, fname, dst, size) for fname in fnames]

    pool = mlc.SuperPool(10)
    pool.map(do_resize, params, description='resize image')

    print('success.')
