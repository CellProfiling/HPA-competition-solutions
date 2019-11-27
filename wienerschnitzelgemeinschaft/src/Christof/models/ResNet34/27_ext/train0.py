import os, sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
from classification_models.resnet.models import ResNet34
import albumentations as A

MODEL_PATH = 'Christof/models/ResNet34/27_ext/'
exp_suffix = '_0'

SIZE = 512

# Load dataset info
path_to_train = 'Christof/assets/train_rgb_512/'
data = pd.read_csv('Christof/assets/train.csv')

normal_aug = A.Compose([#A.Rotate((0,30),p=0.75),
                        A.RandomRotate90(p=1),
                        A.HorizontalFlip(p=0.5),
                        #A.RandomBrightness(0.05),
                        #A.RandomContrast(0.05),
                        A.IAAAffine(translate_percent=10,rotate=45,shear=10, scale=(0.9,1.1)),
                        #A.RandomAffine(degrees=45, translate=(0.1,0.1), shear=10, scale=(0.9,1.1))
                        A.Normalize(mean=(0.08069, 0.05258, 0.05487), std=(0.1300, 0.0879, 0.1386),
                                    max_pixel_value=255.)
                        ])


normal_aug_ext = A.Compose([#A.Rotate((0,30),p=0.75),
                        A.RandomRotate90(p=1),
                        A.HorizontalFlip(p=0.5),
                        #A.RandomBrightness(0.05),
                        #A.RandomContrast(0.05),
                        A.IAAAffine(translate_percent=10,rotate=45,shear=10, scale=(0.9,1.1)),
                        #A.RandomAffine(degrees=45, translate=(0.1,0.1), shear=10, scale=(0.9,1.1))
                        A.Normalize(mean=(0.1174382,  0.06798691, 0.06592218), std=(0.16392466 ,0.10036821, 0.16703453),
                                    max_pixel_value=255.)
                        ])

val_aug = A.Compose([A.HorizontalFlip(p=0.5),
                     A.Normalize(mean=(0.08069, 0.05258, 0.05487), std=(0.1300, 0.0879, 0.1386),
                                 max_pixel_value=255.)])

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

sampling_weights = [   2.6473,   35.0588 ,   8.2069  , 19.3439 ,  16.0145 ,  13.3245 ,  32.8644,
   10.607 ,  551.3   ,  501.1818 , 787.5714 ,  25.8523 ,  39.0301,   51.644,
   30.0846 ,1470.1333 ,  62.8262,  190.1034 ,  39.3084 ,  23.2126 , 170.9457
,    8.2592,   33.2609 ,   9.6889  , 92.2678  ,  4.19 ,    99.3333 ,3150.2857]

sample_weights_ext = [   2.6728,   41.1617 ,  10.3068 ,  42.4172  , 22.9729 ,  21.9808  , 26.8267
,   11.5358 , 474.8659 , 486.7375 , 492.8987  , 66.963 ,   50.2763  , 82.7609,
   45.0683, 1854.2381,  100.3582 , 319.1721  , 76.5762  , 33.424 ,  272.3007,
    7.3664 ,  39.4319  , 10.239 ,  734.6981 ,   2.548 ,  196.6616 , 638.3443]




train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    path = os.path.join(path_to_train, name)
    labs = np.array([int(label) for label in labels])
    bucket_ind = np.argmin([desired[l] for l in labs])
    bucket = labs[bucket_ind]
    weight = sampling_weights[bucket]
    train_dataset_info.append({
        'path': path,
        'labels': labs,
        'weight':weight})
train_dataset_info = np.array(train_dataset_info)

data_ext1 = pd.read_csv('Christof/assets/train_ext1.csv')
path_to_train_ext1 = 'Christof/assets/ext_tomomi/'
train_dataset_info_ext1 = []
for name, labels in zip(data_ext1['Id'], data_ext1['Target'].str.split(' ')):
    path =  os.path.join(path_to_train_ext1, name[:-5])
    labs = np.array([int(label) for label in labels])
    bucket_ind = np.argmin([desired[l] for l in labs])
    bucket = labs[bucket_ind]
    weight = sample_weights_ext[bucket]
    train_dataset_info_ext1.append({
        'path':path,
        'labels': labs,
    'weight':weight})
train_dataset_info_ext1 = np.array(train_dataset_info_ext1)


#rare_dataset_info = np.array([item for item in train_dataset_info if np.isin(item['labels'], rare_classes).any()])
#train_dataset_info = rare_dataset_info

from skimage.morphology import binary_dilation

from classification_models.resnet import preprocess_input
class data_generator:

    @staticmethod
    def create_train(dataset_info, batch_size, shape, augument=None, weighted_sample = True):
        assert shape[2] == 3

        assert shape[2] == 3

        if weighted_sample:
            p = np.array([item['weight'] for item in dataset_info])
            p = p/np.sum(p)
        else:
            p = None

        while True:
            mean_g = 0.05258
            std_g = 0.0879
            #dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                #end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = np.random.choice(dataset_info,batch_size,p=p)
                batch_labels = np.zeros((len(X_train_batch), 28))
                batch_label_maps32 = np.zeros((len(X_train_batch), 32, 32, 28))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(X_train_batch[i]['path'], shape)
                    #image = preprocess_input(image)
                    #rare = np.isin(X_train_batch[i]['labels'], rare_classes).any()

                    if augument:
                        image = data_generator.augment(augument,image)

                    #denormalize
                    img_g = image[:, :, 1].copy()
                    img_g *= np.ones(img_g.shape) * std_g
                    img_g += np.ones(img_g.shape) * mean_g

                    gthresh = 0.12
                    img_g = (img_g > gthresh).astype(float)
                    gr = cv2.resize(img_g, (32, 32), interpolation=cv2.INTER_AREA)
                    gr = binary_dilation(gr).astype(int)
                    for kk in X_train_batch[i]['labels']:
                        batch_label_maps32[i,:, :, kk] = gr

                    batch_images.append(image)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), [batch_labels,batch_label_maps32]


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
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
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

class ZeroOne(Constraint):
    """Constrains the weights to be non-negative.
    """

    def __call__(self, w):
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        w *= K.cast(K.less_equal(w, 1.), K.floatx())
        return w

class NoisyAndPooling2D(Layer):



    #@interfaces.legacy_global_pooling_support
    def __init__(self, data_format=None, **kwargs):
        super(NoisyAndPooling2D, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        self.initializer = initializers.get('zeros')
        self.a = 10

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], input_shape[3])
        else:
            return (input_shape[0], input_shape[1])

    def call(self, inputs):
        a = K.cast(self.a, dtype=K.dtype(inputs))
        P = (K.sigmoid(a*(K.mean(inputs,axis=(1,2))-self.b)) - K.sigmoid(-a * self.b)) / (K.sigmoid(a * (1. - self.b)) - K.sigmoid(-a * self.b))
        return P

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(NoisyAndPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        dim = input_shape[-1]
        shape = (dim,)
        self.b = self.add_weight(shape=shape,
                                    name='b',
                                    initializer=self.initializer,
                                    #regularizer=self.beta_regularizer,
                                    constraint=ZeroOne()
                                    )



def create_model(input_shape, n_out):
    input_tensor = Input(shape=(SIZE, SIZE, 3))
    #bn = BatchNormalization()(input_tensor)
    #conv = Conv2D(3,(3,3),padding='same',activation='relu')(bn)
    base_model = ResNet34(include_top=False,
                             weights='imagenet',
                             input_shape=(SIZE, SIZE, 3),input_tensor=input_tensor)

    x = base_model.get_layer('stage4_unit1_relu1').output

    #x = Dropout(0.5)(x)
    x_map32 = Conv2D(28, kernel_size=(1, 1), activation='sigmoid',name='map32')(x)
    x = NoisyAndPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    #x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)


    output = Dense(n_out,activation='sigmoid',name = 'pred')(x)
    model = Model(input_tensor, [output,x_map32])

    # transfer imagenet weights
    #res_img = ResNet34(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
    #offset = 2
    #for i, l in enumerate(base_model.layers[offset+1:]):
    #    l.set_weights(res_img.layers[i + 1].get_weights())

    return model



# create callbacks list
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras_callbacks import F1Metric, F1MetricField
epochs = 20
batch_size = 16


# split data into train, valid

mskf = MultilabelStratifiedKFold(n_splits=5,shuffle=True,random_state=18)

y = np.zeros((len(train_dataset_info), 28))
for i in range(len(train_dataset_info)):
    y[i][train_dataset_info[i]['labels']] = 1
mskf.get_n_splits(train_dataset_info, y)
kf = mskf.split(train_dataset_info, y)
fold_id = 0
for f in range(fold_id):
    train_indexes, valid_indexes = next(kf)

train_indexes, valid_indexes = next(kf)
train_generator_orig = data_generator.create_train(train_dataset_info[train_indexes],
                                              batch_size, (SIZE, SIZE, 3), augument=normal_aug)
train_generator_ext1 = data_generator.create_train(train_dataset_info_ext1,
                                              batch_size, (SIZE, SIZE, 3), augument=normal_aug_ext)
import random


def gen():
    while True:
        x = random.random()
        if x > 0.5:
            batch = next(train_generator_orig)
        else:
            batch = next(train_generator_ext1)
        yield batch

train_generator = gen()
validation_generator = data_generator.create_train(train_dataset_info[valid_indexes],
                                                   batch_size, (SIZE, SIZE, 3), augument=val_aug, weighted_sample=False)

checkpoint = ModelCheckpoint(MODEL_PATH + 'model_loss{}.h5'.format(exp_suffix), monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)
tensorboard = TensorBoard(MODEL_PATH + 'logs{}'.format(fold_id) + '{}'.format(exp_suffix)  + '/')


nb_epochs = epochs
nb_cycles = 1
init_lr = 0.0005
def _cosine_anneal_schedule(t):

    cos_inner = np.pi * (t % (nb_epochs // nb_cycles))
    cos_inner /= nb_epochs// nb_cycles
    cos_out = np.cos(cos_inner) + 1
    return float(init_lr / 2 * cos_out)

lr_schedule = LearningRateScheduler(_cosine_anneal_schedule,verbose=True)




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
loss_funcs = {
    "pred": custom_loss,
    "map32": weighted_binary_crossentropy
}
loss_weights = {"pred": 1.0, "map32": 5.0 }

metrics = { "pred":[binary_accuracy, brian_f1],
            "map32":[binary_accuracy]}


model.compile(loss=loss_funcs, loss_weights=loss_weights,
              optimizer=Adam(lr=5e-4),
              metrics=metrics)
model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(2* len(train_indexes)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=np.ceil(float(2*len(valid_indexes)) / float(batch_size)),
    epochs=epochs,
    verbose=1,
    callbacks=callbacks_list)

model.load_weights(MODEL_PATH + f'model_loss{exp_suffix}.h5')
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
draw_predict_map32 = np.zeros((len(submit['Id']), 28))
for i, name in tqdm(enumerate(submit['Id'])):
    path = os.path.join('Christof/assets/test_rgb_512/', name)
    image = data_generator.load_image(path, (SIZE, SIZE, 3))



    images = [data_generator.augment(normal_aug, image) for _ in range(tta)]
    tta_predicts, tta_predicts_map32 = model.predict(np.array(images))
    draw_predict[i] = np.median(tta_predicts,axis = 0)

    tta_predicts_map32 = np.percentile(tta_predicts_map32, 95., axis=(1, 2))
    draw_predict_map32[i] = np.median(tta_predicts_map32,axis = 0)


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



submit2 = pd.read_csv('Christof/assets/sample_submission.csv')
# custom thresholds to match lb proportions
thresholds = np.linspace(0.95, 0.05, 101)
pred = draw_predict_map32.copy()
for j in tqdm(range(pred.shape[1])):
    for t in thresholds:
        pred[:, j] = (draw_predict_map32[:, j] > t).astype(int)
        prop = np.mean(pred[:, j])
        if prop >= desired[j]: break
    print(j, '%3.2f' % t, '%6.4f' % desired[j], '%6.4f' % prop, j, )

print(pred[:5].astype(int))

label_predict = [np.arange(28)[score_predict == 1] for score_predict in pred]
str_predict_label = [' '.join(str(l) for l in lp) for lp in label_predict]

submit2['Predicted'] = str_predict_label
# np.save('draw_predict_InceptionV3.npy', score_predict)
submit2.to_csv(MODEL_PATH + 'submission_loss{}_lb_dist_adjusted_map32_8tta.csv'.format(exp_suffix), index=False)


submit3 = pd.read_csv('Christof/assets/sample_submission.csv')
# custom thresholds to match lb proportions
#dp = np.zeros((2,draw_predict.shape[0],draw_predict.shape[1]))
dp = (2*draw_predict + draw_predict_map32) / 3
pred = dp.copy()
thresholds = np.linspace(0.95, 0.05, 101)
applied_threshs = np.zeros(28)
for j in tqdm(range(pred.shape[1])):
    for t in thresholds:
        pred[:, j] = (dp[:, j] > t).astype(int)
        prop = np.mean(pred[:, j])
        if prop >= desired[j]:
            applied_threshs[j] = t
            break
    print(j, '%3.2f' % t, '%6.4f' % desired[j], '%6.4f' % prop, j, )

print(pred[:5].astype(int))

label_predict = [np.arange(28)[score_predict == 1] for score_predict in pred]
str_predict_label = [' '.join(str(l) for l in lp) for lp in label_predict]

submit3['Predicted'] = str_predict_label




# np.save('draw_predict_InceptionV3.npy', score_predict)
submit3.to_csv(MODEL_PATH + 'submission_loss{}_lb_dist_adjusted_mean_loss_map32_8tta.csv'.format(exp_suffix), index=False)




from Christof.utils import f1_sub

sub_best = pd.read_csv('ens56d.csv')
print(f1_sub(sub_best,submit))
print(f1_sub(sub_best,submit2))
print(f1_sub(sub_best,submit3))
s2 = pd.read_csv('Christof/models/GAPNet/13_ext/submission_loss_1_lb_dist_adjusted_8tta.csv')
f1_sub(sub_best,s2)