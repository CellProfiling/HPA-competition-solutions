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
from classification_models.resnet.models import ResNet34
import albumentations as A

from skimage.morphology import binary_dilation


MODEL_PATH = 'Christof/models/ResNet34/20/'
exp_suffix = '_base'

SIZE = 256

# Load dataset info
path_to_train = 'Christof/assets/train_rgb_256/'
data = pd.read_csv('Christof/assets/train.csv')

normal_aug = A.Compose([A.Rotate((0,30),p=0.75),
                        A.RandomRotate90(p=1),
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightness(0.05),
                        A.RandomContrast(0.05),
                        #A.Normalize(mean = (0.08069, 0.05258, 0.05487),std=(0.13704, 0.10145, 0.15313),max_pixel_value=255.)
                        ])



val_aug = A.Compose([A.HorizontalFlip(p=0.5),
                     #A.Normalize(mean = (0.08069, 0.05258, 0.05487),std=(0.13704, 0.10145, 0.15313),max_pixel_value=255.)
                    ])

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
                #batch_labels = np.zeros((len(X_train_batch), 28))
                batch_labels = np.zeros((len(X_train_batch), 32,32,28))
                #gmask = np.zeros((32, 32, 28))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(X_train_batch[i]['path'], shape)
                    #image = preprocess_input(image)
                    #rare = np.isin(X_train_batch[i]['labels'], rare_classes).any()

                    if augument:
                        image = data_generator.augment(augument,image)

                    batch_images.append(image)
                    img_g = image[:, :, 1]/255.

                    gthresh = 0.12
                    img_g = (img_g > gthresh).astype(float)
                    gr = cv2.resize(img_g, (32, 32), interpolation=cv2.INTER_AREA)
                    gr = binary_dilation(gr).astype(int)


                    for kk in X_train_batch[i]['labels']:
                        batch_labels[i,:, :, kk] = gr
                    #batch_labels[i][X_train_batch[i]['labels']] = 1

                yield np.array(batch_images, np.float32), batch_labels

    @staticmethod
    def load_image(path, shape):
        image = cv2.imread(path + '.png', cv2.IMREAD_UNCHANGED)
        return image

    @staticmethod
    def augment(aug,image):
        image_aug = aug(image=image)['image']
        return image_aug


img = data_generator.load_image(train_dataset_info[0]['path'],())
img = data_generator.augment(normal_aug,img)
label = np.zeros((28))
label[train_dataset_info[0]['labels']] = 1








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

def create_model(input_shape, n_out):
    input_tensor = Input(shape=(SIZE, SIZE, 3))
    #bn = BatchNormalization()(input_tensor)
    #conv = Conv2D(3,(3,3),padding='same',activation='relu')(bn)
    base_model = ResNet34(include_top=False,
                             weights='imagenet',
                             input_shape=(SIZE, SIZE, 3),input_tensor=input_tensor)

    x = base_model.get_layer('stage3_unit1_relu1').output

    #x = Dropout(0.5)(x)
    output = Conv2D(28, kernel_size=(1, 1), activation='sigmoid')(x)
    #output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)

    # transfer imagenet weights
    #res_img = ResNet34(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
    #offset = 2
    #for i, l in enumerate(base_model.layers[offset+1:]):
    #    l.set_weights(res_img.layers[i + 1].get_weights())

    return model



# create callbacks list
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras_callbacks import F1MetricField
#from keras_metrics import f1, f1_02
#from keras_losses import  f1_loss
epochs = [1,100]
batch_size = 32


# split data into train, valid

mskf = MultilabelStratifiedKFold(n_splits=5,shuffle=True,random_state=18)

y = np.zeros((len(train_dataset_info), 28))
for i in range(len(train_dataset_info)):
    y[i][train_dataset_info[i]['labels']] = 1
mskf.get_n_splits(train_dataset_info, y)
kf = mskf.split(train_dataset_info, y)
fold_id = 0
train_indexes, valid_indexes = next(kf)

train_generator = data_generator.create_train(train_dataset_info[train_indexes],
                                              batch_size, (SIZE, SIZE, 3), augument=normal_aug,  oversample_factor=0)
validation_generator = data_generator.create_train(train_dataset_info[valid_indexes],
                                                   batch_size, (SIZE, SIZE, 3), augument=val_aug, oversample_factor=0)
validation_generator2 = data_generator.create_train(train_dataset_info[valid_indexes],
                                                   batch_size, (SIZE, SIZE, 3), augument=val_aug, oversample_factor=0)

checkpoint = ModelCheckpoint(MODEL_PATH + 'model_f1all{}.h5'.format(exp_suffix), monitor='val_f1_all', verbose=1,
                             save_best_only=True, mode='max', save_weights_only=True)
checkpoint2 = ModelCheckpoint(MODEL_PATH + 'model_loss{}.h5'.format(exp_suffix), monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)
tensorboard = TensorBoard(MODEL_PATH + 'logs{}'.format(fold_id) + '{}'.format(exp_suffix)  + '/')
# reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
#                                   verbose=1, mode='auto', epsilon=0.0001)
# early = EarlyStopping(monitor="val_loss",
#                      mode="min",
#                      patience=6)
#f1_metric = F1Metric(validation_generator2,2*len(valid_indexes)//batch_size,batch_size,28) #2 times val because of val_aug
f1_metric = F1MetricField(validation_generator2,2*len(valid_indexes)//batch_size,batch_size,28)
#callbacks_list = [f1_metric, checkpoint, checkpoint2,tensorboard]
callbacks_list = [f1_metric, checkpoint2,tensorboard]

# warm up model
model = create_model(
    input_shape=(SIZE, SIZE, 3),
    n_out=28)

x,y = next(train_generator)
y0 = y[0,:,:,:]
np.max(y0,axis = (0,1))
img1 = y0[:,:,7]

for layer in model.layers:
    layer.trainable = False
#model.layers[2].trainable = True
model.layers[-1].trainable = True
model.layers[-2].trainable = True
model.layers[-3].trainable = True
model.layers[-4].trainable = True
model.layers[-5].trainable = True
model.layers[-6].trainable = True


model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-03),
    metrics=['acc'])
# model.summary()
model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
    epochs=epochs[0],
    verbose=1)



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
for layer in model.layers:
    layer.trainable = True
model.compile(loss=weighted_binary_crossentropy,
              optimizer=Adam(lr=1e-3),
              metrics=[binary_accuracy])
model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=2*np.ceil(float(len(valid_indexes)) / float(batch_size)),
    epochs=epochs[1],
    verbose=1,
    callbacks=callbacks_list)


f1_val_bsize = 32
f1_val_steps = 2*len(valid_indexes) // f1_val_bsize
nclasses = 28

preds = np.zeros(shape=(f1_val_steps, f1_val_bsize, 32, 32, nclasses))
y_trues = np.zeros(shape=(f1_val_steps, f1_val_bsize, nclasses))
for s in tqdm(range(f1_val_steps)):
    x, y = next(validation_generator)
    if x.shape[0] == f1_val_bsize:
        pred = np.asarray((model.predict(x)))

        preds[s] = pred
        # vts = np.vstack(y_trues)
        vts_max = np.max(y, axis=(1, 2))
        vts = (vts_max > 0).astype(float)

        y_trues[s] = vts

from sklearn.metrics import f1_score
gpct = 95.
vps = np.vstack(preds)
vts = np.vstack(y_trues)
vpsp = np.percentile(vps, gpct, axis=(1, 2))
thresholds = np.linspace(0, 1, 101)
scores = np.array([f1_score(vts, np.int32(vpsp > t),
                            average='macro') for t in tqdm(thresholds)])
threshold_best_index = np.argmax(scores)
vf1 = scores[threshold_best_index]




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
    path = os.path.join('Christof/assets/test_rgb_256/', name)
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
