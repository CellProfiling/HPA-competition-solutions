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

MODEL_PATH = 'Christof/models/ResNet34/auto/1/'
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
                        ])

val_aug = A.HorizontalFlip(p=0.5)

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
                    np_img = np.array(batch_images, np.float32) /255
                yield np_img, [batch_labels,np_img]

    @staticmethod
    def load_image(path, shape):
        image = cv2.imread(path + '.png', cv2.IMREAD_UNCHANGED)
        return image

    @staticmethod
    def augment(aug,image):
        image_aug = aug(image=image)['image']
        return image_aug

    @staticmethod
    def heavy_augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(scale=(0.5,2.0)),
                iaa.Affine(shear=15),
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=35),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(translate_percent=0.1),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Noop()
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
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

from keras.layers import UpSampling2D
def decoder_block(x,blocks=5,start_filters=512):
    for i in range(blocks):
        x = Conv2D(start_filters//(2**i),(3,3), activation='relu', padding='same')(x)
        x = Conv2D(start_filters//(2**i),(3,3), activation='relu', padding='same')(x)
        x = UpSampling2D()(x)
    return x


def create_model(input_shape, n_out):
    input_tensor = Input(shape=(SIZE, SIZE, 3))
    #bn = BatchNormalization()(input_tensor)
    #conv = Conv2D(3,(3,3),padding='same',activation='relu')(bn)
    base_model = ResNet34(include_top=False,
                             weights='imagenet',
                             input_shape=(SIZE, SIZE, 3),input_tensor=input_tensor)

    x = base_model.output
    x1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    x1 = Flatten()(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    predictions = Dense(n_out, activation='sigmoid',name = 'predictions')(x1)

    x2 = decoder_block(x,5,512)
    img_out = Dense(3, activation='sigmoid', name='img_out')(x2)


    model = Model(input_tensor, [predictions,img_out])

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
epochs = [2,150]
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
checkpoint2 = ModelCheckpoint(MODEL_PATH + 'model_wbce{}.h5'.format(exp_suffix), monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)
tensorboard = TensorBoard(MODEL_PATH + 'logs{}'.format(fold_id) + '{}'.format(exp_suffix)  + '/')
# reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
#                                   verbose=1, mode='auto', epsilon=0.0001)
# early = EarlyStopping(monitor="val_loss",
#                      mode="min",
#                      patience=6)
#f1_metric = F1Metric(validation_generator2,2*len(valid_indexes)//batch_size,batch_size,28) #2 times val because of val_aug
#callbacks_list = [f1_metric, checkpoint, checkpoint2,tensorboard]

callbacks_list = [checkpoint2,tensorboard]

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

def soft_f1_loss(logits, labels):
    __small_value=1e-6
    beta = 1
    #batch_size = logits.size()[0]
    p = logits
    l = labels
    num_pos = K.sum(p, 1) + __small_value
    num_pos_hat = K.sum(l, 1) + __small_value
    tp = K.sum(l * p, 1)
    precise = tp / num_pos
    recall = tp / num_pos_hat
    fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + __small_value)
    loss = K.mean(fs,axis=0)
    return (1 - loss)

from keras.metrics import binary_accuracy
from keras.losses import binary_crossentropy

loss_funcs = {
    "predictions": weighted_binary_crossentropy,
    "img_out": binary_crossentropy
}
loss_weights = {"predictions": 1.0, "img_out": 10.0 }

metrics = { "predictions":[
    binary_accuracy, soft_f1_loss,
]}

# train all layers

for layer in model.layers:
    layer.trainable = True
model.compile(loss=loss_funcs,loss_weights=loss_weights,
              optimizer=Adam(lr=1e-4),
              metrics=metrics)
model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=2*np.ceil(float(len(valid_indexes)) / float(batch_size)),
    epochs=epochs[1],
    verbose=1,
    callbacks=callbacks_list)

model.load_weights(MODEL_PATH + 'model{}.h5'.format(exp_suffix))
preds = np.zeros(shape=(len(valid_indexes),28))
y_true= np.zeros(shape=(len(valid_indexes),28))
for i, info in tqdm(enumerate(train_dataset_info[valid_indexes])):
    image = data_generator.load_image(info['path'], (SIZE, SIZE, 3))
    preds[i] = model.predict(image[np.newaxis])[0]
    y_true[i][info['labels']]=1

from sklearn.metrics import f1_score

individual_f1_scores = np.zeros(28)
for i in range(28):
    individual_f1_scores[i] = f1_score(y_true[:,i],preds[:,i])
individual_f1_scores = pd.DataFrame(individual_f1_scores,columns=['f1'])
individual_f1_scores.to_csv(MODEL_PATH + f'summary_f1_{exp_suffix}.csv',index=False)



f1_res = f1_score(y_true, preds, average='macro')
f1_res_05 = f1_score(y_true, preds_05, average='macro')
print(f1_res)
print(f1_res_05)
SUBMISSION = True

thresholds = np.linspace(0, 1, 1000)
score = 0.0
test_threshold=0.5*np.ones(28)
best_threshold=np.zeros(28)
best_val = np.zeros(28)
for i in range(28):
    for threshold in thresholds:
        test_threshold[i] = threshold
        max_val = np.max(preds)
        val_predict = (preds > test_threshold)
        score = f1_score(y_true > 0.5, val_predict, average='macro')
        if score > best_val[i]:
            best_threshold[i] = threshold
            best_val[i] = score

    print("Threshold[%d] %0.6f, F1: %0.6f" % (i,best_threshold[i],best_val[i]))
    test_threshold[i] = best_threshold[i]

print("Best threshold: ")
print(best_threshold)
print("Best f1:")
print(best_val)



if SUBMISSION:

    submit = pd.read_csv('Christof/assets/sample_submission.csv')
    predicted = []
    draw_predict = []

    for name in tqdm(submit['Id']):
        path = os.path.join('Christof/assets/test_rgb_256/', name)
        image = data_generator.load_image(path, (SIZE, SIZE, 3))
        score_predict = model.predict(image[np.newaxis])[0]
        draw_predict.append(score_predict)

        thresh = max(score_predict[np.argsort(score_predict, axis=-1)[-5]],0.2)
        label_predict = np.arange(28)[score_predict >= thresh]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    submit['Predicted'] = predicted
    #np.save('draw_predict_InceptionV3.npy', score_predict)
    submit.to_csv(MODEL_PATH + 'submission{}_{:.4}.csv'.format(exp_suffix,f1_res), index=False)

    predicted = []
    draw_predict = []

    for name in tqdm(submit['Id']):
        path = os.path.join('Christof/assets/test_rgb_256/', name)
        image = data_generator.load_image(path, (SIZE, SIZE, 3))
        score_predict = model.predict(image[np.newaxis])[0]
        draw_predict.append(score_predict)

        thresh = max(score_predict[np.argsort(score_predict, axis=-1)[-5]],0.5)
        label_predict = np.arange(28)[score_predict >= thresh]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    submit['Predicted'] = predicted
    #np.save('draw_predict_InceptionV3.npy', score_predict)
    submit.to_csv(MODEL_PATH + 'submission{}_{:.4}.csv'.format(exp_suffix,f1_res_05), index=False)

    predicted = []
    draw_predict = []

    for name in tqdm(submit['Id']):
        path = os.path.join('Christof/assets/test_rgb_256/', name)
        image = data_generator.load_image(path, (SIZE, SIZE, 3))
        score_predict = model.predict(image[np.newaxis])[0]
        draw_predict.append(score_predict)

        #thresh = max(score_predict[np.argsort(score_predict, axis=-1)[-5]],0.5)
        label_predict = np.arange(28)[score_predict >= best_threshold]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    submit['Predicted'] = predicted
    #np.save('draw_predict_InceptionV3.npy', score_predict)
    submit.to_csv(MODEL_PATH + 'submission{}_best_val_{:.4}-{:.4}.csv'.format(exp_suffix,best_val[0],best_val[-1]), index=False)

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


    draw_predict = np.zeros((len(submit['Id']),28))

    for i,name in tqdm(enumerate(submit['Id'])):
        path = os.path.join('Christof/assets/test_rgb_256/', name)
        image = data_generator.load_image(path, (SIZE, SIZE, 3))
        draw_predict[i] = model.predict(image[np.newaxis])[0]



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

    # thresh = max(score_predict[np.argsort(score_predict, axis=-1)[-5]],0.5)
    label_predict = [np.arange(28)[score_predict ==1] for score_predict in pred]
    str_predict_label = [' '.join(str(l) for l in lp) for lp in label_predict]

    submit['Predicted'] = str_predict_label
    #np.save('draw_predict_InceptionV3.npy', score_predict)
    submit.to_csv(MODEL_PATH + 'submission{}_lb_dist_adjusted.csv'.format(exp_suffix), index=False)
