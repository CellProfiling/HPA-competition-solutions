from skimage.io import imread, imsave
import os
import cv2
from os.path import join as pjoin, basename, splitext
from lib.utils import info, multiprocessing, debug


def resize_one_img(task):
    fn = task['fn']
    input_dir = task['global_']['input_dir']
    output_dir = task['global_']['output_dir']
    resize_to = task['global_']['resize_to']

    input_path = pjoin(input_dir, fn)
    output_path = pjoin(output_dir, splitext(fn)[0])

    imsave(output_path, imread(input_path))

    return {
        'id_': fn,
    }


def resize():
    input_dir = 'data/valid_windowed'
    resize_to = 'png'
    output_dir = f'data/valid_windowed_{resize_to}'

    os.makedirs(output_dir, exist_ok=True)

    fns = os.listdir(input_dir)
    global_ = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'resize_to': resize_to,
    }

    task_iter = ({'fn': fn, 'global_': global_} for fn in fns)

    multiprocessing(resize_one_img, task_iter, len_=len(fns), n_threads=70)


if __name__ == '__main__':
    resize()
