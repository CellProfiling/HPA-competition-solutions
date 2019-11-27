import os, sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image
import cv2
from sklearn.utils import class_weight, shuffle
from ml_stratifiers import MultilabelStratifiedKFold
import warnings
warnings.filterwarnings("ignore")
from classification_models.resnet.models import ResNet18
import albumentations as A

MODEL_PATH = 'Christof/models/GAPNet/11_tests_on_clr/'
exp_suffix = '_russ_schedule'

SIZE = 512

# Load dataset info
path_to_train = 'Christof/assets/train_rgb_512/'
data = pd.read_csv('Christof/assets/train.csv')

normal_aug = A.Compose([A.Rotate((0,30),p=0.75),
                        A.RandomRotate90(p=1),
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightness(0.05),
                        A.RandomContrast(0.05),
                        A.Normalize(mean=(0.08069, 0.05258, 0.05487), std=(0.1300, 0.0879, 0.1386),
                                    max_pixel_value=255.)
                        ])

val_aug = A.Compose([A.HorizontalFlip(p=0.5),
                     A.Normalize(mean=(0.08069, 0.05258, 0.05487), std=(0.1300, 0.0879, 0.1386),
                                 max_pixel_value=255.)])

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path': os.path.join(path_to_train, name),
        'labels': np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)

counts = np.zeros(28)
for item in train_dataset_info:
    for l in item['labels']:
        counts[l] = counts[l] + 1

counts = counts / len(train_dataset_info)
rare_classes = np.where(counts < 0.005)

#rare_dataset_info = np.array([item for item in train_dataset_info if np.isin(item['labels'], rare_classes).any()])
#train_dataset_info = rare_dataset_info


from classification_models.resnet import preprocess_input
class data_generator:

    @staticmethod
    def create_train(dataset_info, batch_size, shape, augument=None, oversample_factor = 0):
        assert shape[2] == 3



        if oversample_factor > 0:

            rare_dataset_info = np.array([item for item in dataset_info if np.isin(item['labels'], rare_classes).any()])
            #rare_dataset_info = shuffle(rare_dataset_info)
            for i in range(oversample_factor):
            #dataset_info
                dataset_info = np.append(dataset_info,rare_dataset_info)
        while True:
            dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), 28))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(X_train_batch[i]['path'], shape)
                    #image = preprocess_input(image)
                    #rare = np.isin(X_train_batch[i]['labels'], rare_classes).any()

                    if augument:
                        image = data_generator.augment(augument,image)

                    batch_images.append(image)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    @staticmethod
    def load_image(path, shape):
        image = cv2.imread(path + '.png', cv2.IMREAD_UNCHANGED)
        return image

    @staticmethod
    def augment(aug,image):
        image_aug = aug(image=image)['image']
        return image_aug


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, Concatenate, Input, Conv2D
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
import keras
from keras.models import Model


from keras.layers import Layer, InputSpec
from keras import initializers
from keras.constraints import Constraint
import keras.backend as K


from keras.layers import Reshape, Permute, multiply
def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def encoder(backbone):

    c0 = backbone.get_layer('relu0').output

    c1 = backbone.get_layer('stage2_unit1_relu1').get_output_at(0)  # 128
    c2 = backbone.get_layer('stage3_unit1_relu1').output  # 63
    enc_out = backbone.get_layer('stage4_unit1_relu1').output  # 32
    #c4 = backbone.get_layer('stage4_unit1_relu1').output  # 16
    #enc_out = backbone.output  # 8

    short_cuts = [c0,c1,c2]
    return enc_out, short_cuts


def create_model(input_shape, n_out):
    input_tensor = Input(shape=(SIZE, SIZE, 3))
    #bn = BatchNormalization()(input_tensor)
    #conv = Conv2D(3,(3,3),padding='same',activation='relu')(bn)
    base_model = ResNet18(include_top=False,
                             weights='imagenet',
                             input_shape=(SIZE, SIZE, 3),input_tensor=input_tensor)

    enc_out, short_cuts = encoder(base_model)
    x0 = GlobalAveragePooling2D()(squeeze_excite_block(enc_out))
    x1 = GlobalAveragePooling2D()(squeeze_excite_block(short_cuts[0]))
    x2 = GlobalAveragePooling2D()(squeeze_excite_block(short_cuts[1]))
    x3 = GlobalAveragePooling2D()(squeeze_excite_block(short_cuts[2]))
    x = Concatenate()([x0,x1,x2,x3])
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(n_out, activation='sigmoid')(x)

    model = Model(input_tensor, output)

    # transfer imagenet weights
    #res_img = ResNet34(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
    #offset = 2
    #for i, l in enumerate(base_model.layers[offset+1:]):
    #    l.set_weights(res_img.layers[i + 1].get_weights())

    return model



# create callbacks list
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras_callbacks import F1Metric
#from keras_metrics import f1, f1_02
#from keras_losses import  f1_loss
epochs = [40,150]
batch_size = 32

p = {}
p['lr'] = 1e-4  # Learning rate
p['step_size'] = 5
p['gamma'] = 0.5
p['wd'] = 1e-4  # Weight decay
p['momentum'] = 0.9  # Momentum
p['epoch_size'] = 15 # How many epochs to change learning rate
p['patience'] = 30 # epochs to wait for early stopping

# split data into train, valid

mskf = MultilabelStratifiedKFold(n_splits=5,shuffle=True,random_state=18)

y = np.zeros((len(train_dataset_info), 28))
for i in range(len(train_dataset_info)):
    y[i][train_dataset_info[i]['labels']] = 1
mskf.get_n_splits(train_dataset_info, y)
kf = mskf.split(train_dataset_info, y)
fold_id = 1
for f in range(fold_id):
    train_indexes, valid_indexes = next(kf)

train_indexes, valid_indexes = next(kf)
train_generator = data_generator.create_train(train_dataset_info[train_indexes],
                                              batch_size, (SIZE, SIZE, 3), augument=normal_aug,  oversample_factor=0)
validation_generator = data_generator.create_train(train_dataset_info[valid_indexes],
                                                   batch_size, (SIZE, SIZE, 3), augument=val_aug, oversample_factor=0)
#validation_generator2 = data_generator.create_train(train_dataset_info[valid_indexes],
#                                                   batch_size, (SIZE, SIZE, 3), augument=val_aug, oversample_factor=0)

#checkpoint = ModelCheckpoint(MODEL_PATH + 'model_f1all{}.h5'.format(exp_suffix), monitor='val_f1_all', verbose=1,
#                             save_best_only=True, mode='max', save_weights_only=True)
checkpoint = ModelCheckpoint(MODEL_PATH + 'model_loss{}.h5'.format(exp_suffix), monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)
tensorboard = TensorBoard(MODEL_PATH + 'logs{}'.format(fold_id) + '{}'.format(exp_suffix)  + '/')
# reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
#                                   verbose=1, mode='auto', epsilon=0.0001)
# early = EarlyStopping(monitor="val_loss",
#                      mode="min",
#                      patience=6)
#f1_metric = F1Metric(validation_generator2,2*len(valid_indexes)//batch_size,batch_size,28) #2 times val because of val_aug


from torch.optim.lr_scheduler import StepLR


base_lr = 1e-4
gamma = 0.5
step_size = 5

#base_lr * gamma ** (t // step_size)
def step_schedule(t):
    lr = base_lr * gamma ** (t // step_size)
    return float(lr)

lr_schedule = LearningRateScheduler(step_schedule,verbose=True)

callbacks_list = [lr_schedule, checkpoint, tensorboard]


# warm up model
model = create_model(
    input_shape=(SIZE, SIZE, 3),
    n_out=28)



POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb


def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor
    and a target tensor. POS_WEIGHT is used as a multiplier
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    #_epsilon = K.epsilon()
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)

import tensorflow as tf
from tensorflow.python.framework import ops
from functools import reduce

def binaryRound(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name)

        # For Tensorflow v0.11 and below use:
        #with g.gradient_override_map({"Floor": "Identity"}):
        #    return tf.round(x, name=name)

def brian_f1(y_true, y_pred):
    y_pred = binaryRound(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def brian_f1_loss(y_true, y_pred):
    return 1- brian_f1(y_true, y_pred)


def custom_loss(y_true, y_pred):

    return 4*weighted_binary_crossentropy(y_true,y_pred) - K.log(brian_f1(y_true,y_pred))

# train all layers
from keras.metrics import binary_accuracy

from keras_optimizers import AdamW

model.compile(loss=custom_loss,
              optimizer=AdamW(lr=1e-4,weight_decay=1e-4,batch_size=32,
                              samples_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
                              epochs=epochs[0]),
              metrics=[binary_accuracy,brian_f1])
model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=2*np.ceil(float(len(valid_indexes)) / float(batch_size)),
    epochs=epochs[0],
    verbose=1,
    callbacks=callbacks_list)
model.load_weights(MODEL_PATH + 'model_loss{}.h5'.format(exp_suffix))
eps = 0.004
desired = {
    0: 0.36239782,
    1: 0.043841336,
    2: 0.075268817,
    3: 0.059322034,
    4: 0.075268817,
    5: 0.075268817,
    6: 0.043841336,
    7: 0.075268817,
    8: eps,
    9: eps,
    10: eps,
    11: 0.043841336,
    12: 0.043841336,
    13: 0.014198783,
    14: 0.043841336,
    15: eps,
    16: 0.028806584,
    17: 0.014198783,
    18: 0.028806584,
    19: 0.059322034,
    20: eps,
    21: 0.126126126,
    22: 0.028806584,
    23: 0.075268817,
    24: eps,
    25: 0.222493888,
    26: 0.028806584,
    27: eps
}

submit = pd.read_csv('Christof/assets/sample_submission.csv')
tta = 8



draw_predict = np.zeros((len(submit['Id']), 28))

for i, name in tqdm(enumerate(submit['Id'])):
    path = os.path.join('Christof/assets/test_rgb_512/', name)
    image = data_generator.load_image(path, (SIZE, SIZE, 3))
    images = [data_generator.augment(normal_aug, image) for _ in range(tta)]
    tta_predicts = model.predict(np.array(images))
    draw_predict[i] = np.median(tta_predicts,axis = 0)



# custom thresholds to match lb proportions
thresholds = np.linspace(0.95, 0.05, 101)
pred = draw_predict.copy()
for j in tqdm(range(pred.shape[1])):
    for t in thresholds:
        pred[:, j] = (draw_predict[:, j] > t).astype(int)
        prop = np.mean(pred[:, j])
        if prop >= desired[j]: break
    print(j, '%3.2f' % t, '%6.4f' % desired[j], '%6.4f' % prop, j, )

print(pred[:5].astype(int))

label_predict = [np.arange(28)[score_predict == 1] for score_predict in pred]
str_predict_label = [' '.join(str(l) for l in lp) for lp in label_predict]

submit['Predicted'] = str_predict_label
# np.save('draw_predict_InceptionV3.npy', score_predict)
submit.to_csv(MODEL_PATH + 'submission_loss{}_lb_dist_adjusted_8tta.csv'.format(exp_suffix), index=False)

from Christof.utils import f1_sub

best_sub = pd.read_csv('ens18.csv')
f1_sub(best_sub,submit)