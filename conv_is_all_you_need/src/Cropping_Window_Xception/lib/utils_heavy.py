from os.path import join as pjoin

import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from lib.utils import debug, load_image, gen_cwd_slash
from keras.models import model_from_yaml

from .utils import chunk


def focal_loss(y_true, y_pred):
    gamma = 2.
    y_pred = tf.cast(y_pred, tf.float32)

    max_val = K.clip(-y_pred, 0, 1)
    loss = y_pred - y_pred * y_true + max_val + K.log(K.exp(-max_val) + K.exp(-y_pred - max_val))
    invprobs = tf.log_sigmoid(-y_pred * (y_true * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss

    return K.mean(K.sum(loss, axis=1))


def macro_f1(y_true, y_pred, debug='', epsilon=1e-7):
    # DEBUG
    # y_true = tf.Print(y_true, message=f'\n({debug}) y_true = ', data=[tf.reduce_sum(y_true, axis=0)], summarize=9999)
    # y_pred = tf.Print(y_pred, message=f'\n({debug}) y_pred = ', data=[tf.reduce_sum(y_pred, axis=0)], summarize=9999)

    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    # tn = tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)

    out = 2 * p * r / (p + r + epsilon)
    # replace all NaN's with 0's
    out = tf.where(tf.is_nan(out), tf.zeros_like(out), out)
    out = tf.reduce_mean(out)

    return out


# TODO: why is val_macro_f1_metric not giving the same value as the analyze.py?
def gen_macro_f1_metric(config, debug=''):

    def macro_f1_metric(y_true, y_pred):
        y_pred = tf.where(y_pred >= config['score_threshold'], tf.ones_like(y_pred), tf.zeros_like(y_pred))
        return macro_f1(y_true, y_pred, debug=debug)

    return macro_f1_metric


def model_from_config(config, which='latest', **kwargs):
    cwd_slash = gen_cwd_slash(config)
    with open(cwd_slash('model.yaml'), 'r') as f:
        model = model_from_yaml(f.read())
    model.load_weights(cwd_slash(f'{which}.weights'))

    return model


def predict(anno, config, model=None, batch_size=None, folder=None, extension=None, to_npy=None):
    if model is None:
        debug(f'predict(): loading model from config')
        model = model_from_config(config)

    if batch_size is None:
        debug(f"predict(): loading predict_batch_size = ({config['predict_batch_size']}) from config")
        batch_size = config['predict_batch_size']

    if type(anno) is str:
        debug(f'predict(): loading anno from {anno}')
        anno = pd.read_csv(anno, index_col=0)

    scores_list = []
    chunks = list(chunk(len(anno), batch_size))
    for i_begin, i_end in tqdm(chunks):
        img_list = []
        for id_, row in anno.iloc[i_begin:i_end].iterrows():
            img_list.append(
                load_image(
                    id_,
                    config,
                    channel=None,
                    folder=folder or row.get('folder'),
                    extension=extension or row.get('extension'),
                ) / 255.
            )
        imgs = np.array(img_list)
        scores = model.predict(imgs)
        scores_list.append(scores)

    scores_mat = np.concatenate(scores_list)
    if to_npy is not None:
        np.save(to_npy, scores_mat)
    return scores_mat
