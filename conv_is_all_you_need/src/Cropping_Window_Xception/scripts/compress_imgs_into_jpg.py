import pandas as pd
import PIL
from os import makedirs
from multiprocessing import Pool
from os.path import join as pjoin
from lib.utils import debug, str_to_labels, load_image
from lib.config import gen_config
from skimage.io import imsave

makedirs(output_folder, exist_ok=True)


def compress_one(task):
    id_ = task['id_']
    config = task['config']
    folder = task['folder']
    output_folder = task['output_folder']
    img = load_image(id_, config, resize=False, folder=folder, extension='tif', channel=['red', 'green', 'blue'])
    saved_path = pjoin(output_folder, f'{id_}.jpg')
    imsave(saved_path, img)
    return {'id_': id_, 'saved_path': saved_path}


def compress_task_gen(anno, folder, output_folder):
    config = gen_config()
    for i_row, (id_, row) in enumerate(anno.iterrows()):
        yield {
            'id_': id_,
            'config': config,
            'folder': folder,
            'output_folder': output_folder,
        }


def compress(
    n_threads=70,
    folder='./data/test_full_size',
    output_folder='./data/test_full_size_compressed',
):
    anno = pd.read_csv('./data/sample_submission.csv', index_col=0)
    with Pool(n_threads) as p:
        result_iter = p.imap_unordered(compress_one, compress_task_gen(anno, folder, output_folder))
        for i_result, result in enumerate(result_iter):
            debug(f"({i_result}/{len(anno)}) {result['id_']}")


if __name__ == '__main__':
    compress(folder='./data/train_full_size', output_folder='./data/train_full_size_compressed')
    compress(folder='./data/test_full_size', output_folder='./data/test_full_size_compressed')
