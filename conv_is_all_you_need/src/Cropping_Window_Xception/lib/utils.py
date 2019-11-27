import json
import os
import math
import random
import sys
import redis
import pickle
from itertools import chain, cycle, islice
from multiprocessing import Pool
from os.path import join as pjoin

import colors
import cv2
import GPUtil
import matplotlib.pyplot as plt
import functools
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from PIL import Image


def roundrobin(iterables):
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def take(n, iterable):
    return list(islice(iterable, n))


def flatten(ls):
    return list(chain.from_iterable(ls))


def gen_cwd_slash(config):

    def cwd_slash(*args):
        return pjoin(config['_cwd'], *args)

    return cwd_slash


def load_config(cwd):
    debug(f"Loading config from {pjoin(cwd, 'config.json')} ... ", end='')
    with open(pjoin(cwd, 'config.json'), 'r') as f:
        config = json.load(f)

    config['_cwd'] = cwd
    debug(f"done")
    return config


def pavel_augment(image):
    iaa.Compose(
        [
            iaa.RandomRotate90(),
            iaa.Flip(),
            iaa.Transpose(),
            iaa.OneOf([
                iaa.IAAAdditiveGaussianNoise(),
                iaa.GaussNoise(),
            ], p=0.2),
            iaa.OneOf([
                iaa.MotionBlur(p=.2),
                iaa.MedianBlur(blur_limit=3, p=.1),
                iaa.Blur(blur_limit=3, p=.1),
            ], p=0.2),
            iaa.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=.5),
            iaa.OneOf([
                iaa.OpticalDistortion(p=0.3),
                iaa.GridDistortion(p=.1),
                iaa.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            iaa.OneOf([
                iaa.IAASharpen(),
                iaa.IAAEmboss(),
                iaa.RandomBrightnessContrast(),
            ], p=0.3)
        ],
        p=1
    )


# TODO: Shouldn't augment return *all* results instead of *one result*?
def augment(image):
    augment_img = iaa.Sequential(
        [
            iaa.OneOf(
                [
                    iaa.Noop(),
                    iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270),
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                ]
            ),
            # iaa.GammaContrast(gamma=(0.85, 1.15), per_channel=True),
        ],
        random_order=True
    )

    image_aug = augment_img.augment_image(image)
    return image_aug


def randomize_and_loop(anno):
    while True:
        anno = anno.sample(frac=1)
        for x in anno.iterrows():
            yield x


def batching_row_gen(row_gen, batch_size):
    while True:
        ids = []
        rows = []
        for _ in range(batch_size):
            id_, row = next(row_gen)
            ids.append(id_)
            rows.append(row)
        yield pd.DataFrame.from_records(rows, index=ids)


def gen_even_batches(anno, config, target_col='Target'):
    labels = [str_to_labels(t) for t in anno[target_col]]

    generators = []
    for class_id in config['class_ids']:
        filtered_anno = anno.loc[[class_id in x for x in labels]]
        if len(filtered_anno) > 0:
            generators.append(randomize_and_loop(filtered_anno))

    # also add a generator for images that have none of the classes in the list
    # filtered_anno = anno.loc[[all([class_id not in x for class_id in config['class_ids']]) for x in labels]]
    # if len(filtered_anno) > 0:
    #     generators.append(randomize_and_loop(filtered_anno))
    #     generators.append(randomize_and_loop(filtered_anno))
    #     generators.append(randomize_and_loop(filtered_anno))

    master_generator = roundrobin(generators)

    return batching_row_gen(master_generator, config['batch_size'])


# this is a generator transformer
#    gen [ batch_anno_df ] -> gen [ (data_mat, label_vec) ]
def data_gen_from_anno_gen(
    anno_gen,
    config,
    group=None,
    target_col=None,
    do_augment=True,
):
    while True:
        anno_batch_df = next(anno_gen)
        batch_x_list = []
        batch_y_list = []
        for id_, row in anno_batch_df.iterrows():
            source_img_size = config['source_img_size']
            window = row[['i_left', 'i_top', 'i_right', 'i_bottom']].copy()

            if do_augment:
                wj = config['aug_window_jittering']
                if wj > 0:
                    h_shift, v_shift = np.random.randint(-wj, wj + 1, 2)
                    h_shift = np.clip(h_shift, -window['i_left'], source_img_size - window['i_right'])
                    v_shift = np.clip(v_shift, -window['i_top'], source_img_size - window['i_bottom'])
                    window[['i_left', 'i_right']] += h_shift
                    window[['i_top', 'i_bottom']] += v_shift

            img = load_windowed_img(
                row['source_img_id'],
                window=window,
                resize=source_img_size,
                channels=config['channels'],
                group=group or row['group'],
            )

            # TODO: augmentation should be on the original resolution (512 x 512) instead of the 299 x 299
            # TODO: augmentation should be heavier on the rare classes
            # TODO: need a way to better pick out windows that actually *has* centrosomes/MOC

            if do_augment and random.random() < config['aug_negative_control']:
                img[:, :, 1] = 0
                y_vec = np.zeros_like(config['class_ids'])
            else:
                if do_augment:
                    img = augment(img)

                if type(target_col) is str:
                    labels = str_to_labels(row[target_col])
                    y_vec = np.array([1 if class_id in labels else 0 for class_id in config['class_ids']])
                else:
                    y_vec = np.zeros_like(config['class_ids'])

            batch_y_list.append(y_vec)
            batch_x_list.append(img / 255.)

        batch_x = np.array(batch_x_list, dtype=np.float32)
        batch_y = np.array(batch_y_list, dtype=np.float32)

        yield batch_x, batch_y


# # TODO: EXP: show some examples of rotation to see if tessellation is needed
# def create_generator_from_anno(anno, folder, config, do_augment=True, do_shuffle=True):
#     while True:
#         if do_shuffle:
#             anno = anno.sample(len(anno))
#         for start in range(0, len(anno), config['batch_size']):
#             end = min(start + config['batch_size'], len(anno))
#             batch_images = []
#             X_train_batch = anno.iloc[start:end]
#             batch_labels = np.zeros((len(X_train_batch), 28))
#             for i in range(len(X_train_batch)):
#                 image = load_img(X_train_batch.iloc[i].name, config, folder=folder)
#                 if do_augment:
#                     # TODO: augmentation should be on the original resolution (512 x 512) instead of the 299 x 299
#                     image = augment(image)
#                 batch_images.append(image / 255.)
#                 targets = str_to_labels(X_train_batch.iloc[i]['Target'])
#                 batch_labels[i][targets] = 1
#             yield np.array(batch_images, np.float32), batch_labels


def labels_to_str(labels, delimiter=' '):
    return delimiter.join(str(x) for x in labels)


def chart(xs, n=10, c='#', y_range=None):
    if y_range is None:
        if max(xs) == 0:
            factor = math.nan
        else:
            factor = 1 / max(xs) * n
    else:
        factor = 1 / y_range * n

    ys = []
    for x in xs:
        tmp = x * factor
        if math.isnan(tmp):
            ys.append('NaN')
        else:
            ys.append(c * int(round(x * factor)))

    return ys


def str_to_labels(str_, delimiter=' '):
    if type(str_) is not str:
        return []
    elif str_ == '':
        return []
    else:
        return list(set(int(x) for x in str_.split(delimiter)))


def eprint(*args, fg='green', bg='black', **kwargs):
    print(*(colors.color(str(x), fg=fg, bg=bg) for x in args), file=sys.stderr, **kwargs)


def debug(*args, **kwargs):
    eprint(*args, fg='green', bg='black', **kwargs)


def info(*args, **kwargs):
    eprint(*args, fg='blue', bg='black', **kwargs)


def warn(*args, **kwargs):
    eprint(*args, fg='yellow', bg='black', **kwargs)


class RedisCached:

    def __init__(self, fn):
        self.fn = fn
        self.cache_connection = redis.Redis()
        self.n_hits = 0
        self.n_calls = 0

    def cache_info(self):
        dbsize = self.cache_connection.dbsize()
        return f"{self.n_hits}/{self.n_calls} ({dbsize})"

    def reset_cache_info(self):
        self.n_hits = 0
        self.n_calls = 0

    def __call__(self, id_, *args, **kwargs):
        self.n_calls += 1
        cached = self.cache_connection.get(id_)
        if cached is not None:
            self.n_hits += 1
            return pickle.loads(cached)

        new_return = self.fn(id_, *args, **kwargs)
        self.cache_connection.set(id_, pickle.dumps(new_return))
        return new_return


def load_img_separate(id_, folder, extension, channels):
    channel_imgs = []
    for channel in channels:
        path_to_img = pjoin(folder, f'{id_}_{channel}{extension}')
        img = np.array(Image.open(path_to_img))
        channel_imgs.append(img)

    return np.stack(channel_imgs, axis=-1).astype(np.uint8)


def load_img_combined(id_, folder, extension):
    path_to_img = pjoin(folder, f'{id_}{extension}')
    img = np.array(Image.open(path_to_img))
    return np.array(img).astype(np.uint8)


def load_img_separate_with_color(id_, folder, extension, channels):
    # TODO: note that the combined rgb jpegs seem to have better resolution than the separated version
    channel_imgs = []
    for channel in channels:
        path_to_img = pjoin(folder, f'{id_}_{channel}{extension}')
        img = np.array(Image.open(path_to_img))
        if channel == 'red':
            img = img[:, :, 0]
        elif channel == 'green':
            img = img[:, :, 1]
        elif channel == 'blue':
            img = img[:, :, 2]
        elif channel == 'yellow':
            img = img[:, :, [0, 1]].mean(axis=2).round().astype(np.uint8)
        channel_imgs.append(img)

    return np.stack(channel_imgs, axis=-1).astype(np.uint8)


def load_windowed_img(source_img_id, window, resize=None, group=None, channels=['red', 'green', 'blue', 'yellow']):
    left, top, right, bottom = window
    img = load_img(source_img_id, resize, group, channels)
    return img[top:bottom, left:right, :]


@RedisCached
def load_img(
    id_,
    resize=None,
    group=None,
    channels=['red', 'green', 'blue', 'yellow'],
):

    if group == 'train':
        img = load_img_separate(id_, 'data/train', '.png', channels)
    elif group == 'train_full_size':
        img = load_img_separate(id_, 'data/train_full_size', '.tif', channels)
    elif group == 'train_full_size_compressed':
        img = load_img_combined(id_, 'data/train_full_size_compressed', '.jpg')
    elif group == 'test':
        img = load_img_separate(id_, 'data/test', '.png', channels)
    elif group == 'test_full_size':
        img = load_img_separate(id_, 'data/test_full_size', '.tif', channels)
    elif group == 'test_full_size_compressed':
        img = load_img_combined(id_, 'data/test_full_size_compressed', '.jpg')
    elif group == 'hpa_rgby':
        img = load_img_separate_with_color(id_, 'data/hpa_public_imgs_rgby', '.jpg', channels)

    if type(resize) is int:
        img = cv2.resize(img, (resize, resize))
    elif type(resize) is tuple:
        img = cv2.resize(img, resize)

    if len(img.shape) == 2:
        img = np.array([img]).transpose([1, 2, 0])

    return img


def cut_score(anno, scores_mat, config, prediction_col='Predicted', score_threshold=None, to_csv=None):
    if type(anno) is str:
        debug(f'loading anno from {anno}')
        anno = pd.read_csv(anno, index_col=0)

    if type(scores_mat) is str:
        debug(f'loading scores_mat from {scores_mat}')
        scores_mat = np.load(scores_mat)

    if score_threshold is None:
        debug(f'score_threshold is not supplied, using the following default from config:')
        debug(f"score_threshold = {config['score_threshold']}")
        score_threshold = config['score_threshold']

    label_predict_list = [np.array(config['class_ids'])[x] for x in scores_mat >= score_threshold]

    predicted_anno = anno.copy()
    predicted_anno[prediction_col] = [labels_to_str(x) for x in label_predict_list]

    if to_csv is not None:
        predicted_anno.to_csv(to_csv, index=True)
        debug(f'saved csv to {to_csv}')

    return predicted_anno


def display_imgs(
    img_list,
    n_cols=None,
    aspect_ratio=1.75,
    save_as=None,
    dpi=100,
    background_color=None,
    text_color='white',
    text_shadow_color='black'
):
    n_imgs = len(img_list)
    if n_cols is None:
        n_cols = math.floor(math.sqrt(n_imgs * aspect_ratio) + (1 / (aspect_ratio + 1)))
    n_rows = math.ceil(n_imgs / n_cols)

    fig = plt.figure(figsize=(n_cols * 10, n_rows * 10))
    for idx, (img, label) in enumerate(img_list):
        fig.add_subplot(n_rows, n_cols, idx + 1)
        plt.axis('off')
        if img.max() > 1 and not issubclass(img.dtype.type, np.integer):
            warn(f'Warning: the image #{idx} has max value {img.max()} but dtype {img.dtype.type}. label = {label}')
        if len(img.shape) == 3:
            if img.shape[2] != 3:
                warn(f'Warning: the image #{idx} has {img.shape[2]} channels. label = {label}')
            plt.imshow(img[:, :, :3])
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        plt.text(
            0.011,
            0.989,
            label,
            fontsize=16,
            color=text_shadow_color,
            transform=plt.gca().transAxes,
            horizontalalignment='left',
            verticalalignment='top',
        )
        plt.text(
            0.01,
            0.99,
            label,
            fontsize=16,
            color=text_color,
            transform=plt.gca().transAxes,
            horizontalalignment='left',
            verticalalignment='top',
        )
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.01, hspace=0.01)

    if save_as is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save_as, dpi=dpi, facecolor=background_color)
        plt.close()
        debug(f'Saved as {save_as}')


def get_n_gpus():
    debug('Detecting the number of GPUs ...')
    n_gpus = len(GPUtil.getGPUs())
    debug(f'Found {n_gpus} GPU(s).')
    return n_gpus


def np_macro_f1(y_true, y_pred, config, epsilon=1e-7, return_details=False):
    details = pd.DataFrame(index=config['class_ids'])
    details['true_positives'] = tp = np.sum(y_true * y_pred, axis=0)
    # details['true_negatives'] = tn = np.sum((1 - y_true) * (1 - y_pred), axis=0)
    details['false_positives'] = fp = np.sum((1 - y_true) * y_pred, axis=0)
    details['false_negatives'] = fn = np.sum(y_true * (1 - y_pred), axis=0)

    details['precision'] = p = tp / (tp + fp + epsilon)
    details['recall'] = r = tp / (tp + fn + epsilon)

    out = 2 * p * r / (p + r + epsilon)
    # replace all NaN's with 0's
    details['f1_score'] = out = np.where(np.isnan(out), np.zeros_like(out), out)
    out = np.mean(out)
    if return_details:
        return (out, details)
    else:
        return out


def format_macro_f1_details(details, config):
    return pd.DataFrame(
        {
            'organelle': pd.Series(config['class_labels'])[details.index],
            'f1': details['f1_score'].round(4),
            'f1_chart': chart(details['f1_score'], n=15, y_range=1),
            'p': details['precision'].round(4),
            'p_chart': chart(details['precision'], n=10, y_range=1),
            'r': details['recall'].round(4),
            'r_chart': chart(details['recall'], n=10, y_range=1),
            'tp': details['true_positives'].round(2),
            'fp': details['false_positives'].round(2),
            'fn': details['false_negatives'].round(2),
            'tpfn': (details['true_positives'] + details['false_negatives']).round(2),
            'tpfn_chart': chart(details['true_positives'] + details['false_negatives'], 20),
        },
        index=details.index,
    )


def preview_generator(g, config, out_folder='./tmp', filename_prefix='preview_generator', n_batches=1):
    for i_batch in range(n_batches):
        batch_x, batch_y = next(g)
        img_list = []
        for i_x, x in enumerate(batch_x):
            img = x[:, :, :3]
            class_ids = np.array(config['class_ids'])[np.nonzero(batch_y[i_x])[0]]
            img_label = class_ids_to_label(class_ids, config)
            img_list.append((img, img_label))
        display_imgs(
            img_list,
            save_as=pjoin(out_folder, f'{filename_prefix}_{i_batch}.png'),
        )


def anno_to_binary(anno, config, target_col='Target', pred_col='Predicted'):
    true_vec_list = []
    pred_vec_list = []
    for i, (idx, row) in enumerate(anno.iterrows()):
        true_vec_list.append(
            np.array([class_id in str_to_labels(row[target_col]) for class_id in config['class_ids']]).astype(int)
        )
        pred_vec_list.append(
            np.array([class_id in str_to_labels(row[pred_col]) for class_id in config['class_ids']]).astype(int)
        )
    true_mat = np.array(true_vec_list)
    pred_mat = np.array(pred_vec_list)

    return true_mat, pred_mat


def class_id_to_label(class_id, config):
    return f"{class_id}-{config['class_labels'][class_id]}"


def class_ids_to_label(class_ids, config):
    return ', '.join([class_id_to_label(class_id, config) for class_id in class_ids])


def chunk(n, c):
    nc = n // c
    for i in range(nc):
        yield (i * c, (i + 1) * c)
    if nc * c < n:
        yield (nc * c, n)


def chunk_df(df, chunk_size):
    n_rows = len(df)
    for i_begin, i_end in chunk(n_rows, chunk_size):
        yield df.iloc[i_begin:i_end]


def banner(message, divider='-', width=80):
    info(divider * width)
    info('\n'.join([m.center(width) for m in message.split('\n')]))
    info(divider * width)


def combine_windows(windowed_anno, min_n_windows, config, save_combined_anno_to=None, group_col='source_img_id'):
    if type(windowed_anno) is str:
        windowed_anno = pd.read_csv(windowed_anno, index_col=0, dtype={'Predicted': object})

    ids = []
    predictions = []
    for id_, group in windowed_anno.groupby(group_col):
        ids.append(id_)
        list_pred = []
        for id_, row in group.iterrows():
            class_ids = str_to_labels(row['Predicted'])
            vec_pred = np.array([class_id in class_ids for class_id in config['class_ids']])
            list_pred.append(vec_pred)
        mat_pred = np.array(list_pred)
        predictions.append(labels_to_str(np.array(config['class_ids'])[mat_pred.sum(axis=0) > min_n_windows]))

    combined_anno = pd.DataFrame({'Id': ids, 'Predicted': predictions})

    if save_combined_anno_to is not None:
        combined_anno.to_csv(save_combined_anno_to, index=False)

    return combined_anno


# def gen_cosine_annealing(period=10, upper=1e-3, lower=0):

#     def cosine_annealing(i_epoch, lr):
#         return (math.cos(math.pi * (i_epoch % period) / period) +
#                 1) / 2 * (upper - lower) * (math.pow(0.5, i_epoch // period)) + lower

#     return cosine_annealing


def multiprocessing(func, iter_, len_=None, n_threads=None, disabled=False):
    if len_ is None:
        if hasattr(iter_, '__len__'):
            len_ = len(iter)
        else:
            len_ = 'unknown_len'
    if disabled:
        for i, x in enumerate(iter_):
            result = func(x)
            info(f"({i}/{len_}) {result.get('id_', 'unknown_id')}")
            yield result
    else:
        with Pool(n_threads) as p:
            results_iter = p.imap(func, iter_)
            for i, result in enumerate(results_iter):
                info(f"({i}/{len_}) {result.get('id_', 'unknown_id')}")
                yield result


class ChunkIter:

    def __init__(self, iterable, chunk_size):
        self.iterable = iterable
        self.chunk_size = chunk_size

    def __iter__(self):
        for _ in range(self.chunk_size):
            yield next(self.iterable)

    def __len__(self):
        return self.chunk_size


def compute_i_coords(in_anno, config):
    anno = in_anno.copy()
    s = config['source_img_size']
    anno['i_left'] = (s * anno['left']).round().clip(0, config['source_img_size'] - config['size']).astype(int)
    anno['i_top'] = (s * anno['top']).round().clip(0, config['source_img_size'] - config['size']).astype(int)
    anno['i_right'] = anno['i_left'] + config['size']
    anno['i_bottom'] = anno['i_top'] + config['size']
    return anno


def vec_to_str(vec):
    return labels_to_str(np.nonzero(np.array(vec).astype(int))[0])
