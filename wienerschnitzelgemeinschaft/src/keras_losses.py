from keras_metrics import f1, f1b
import keras.backend as K
import tensorflow as tf

def f1_loss(y_true, y_pred):
    return 1 - K.mean(f1(y_true, y_pred))

def f1b_loss(y_true, y_pred):
    return 1-K.mean(f1b(y_true, y_pred))




def KerasFocalLoss(target, input):
    """
    Should be applied without sigmoid activtion layer
    from https://www.kaggle.com/rejpalcz/focalloss-for-keras
    :param target:
    :param input:
    :return:
    """
    gamma = 2.
    input = tf.cast(input, tf.float32)

    max_val = K.relu(-input)
    loss = input - input * target + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))
    invprobs = tf.log_sigmoid(-input * (target * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss

    return K.mean(K.sum(loss, axis=1))

