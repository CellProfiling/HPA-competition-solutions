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

MODEL_PATH = 'models/ResNet34/tests/bojan_1/'
SIZE = 256

# Load dataset info
path_to_train = 'assets/train/'
data = pd.read_csv('assets/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path': os.path.join(path_to_train, name),
        'labels': np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)

from classification_models.resnet import preprocess_input
class data_generator:

    @staticmethod
    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), 28))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(
                        X_train_batch[i]['path'], shape)
                    #image = preprocess_input(image)
                    if augument:
                        image = data_generator.augment(image)
                    batch_images.append(image)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    @staticmethod
    def load_image(path, shape):
        image_red_ch = Image.open(path + '_red.png')
        image_yellow_ch = Image.open(path + '_yellow.png')
        image_green_ch = Image.open(path + '_green.png')
        image_blue_ch = Image.open(path + '_blue.png')
        image = np.stack((
            np.array(image_red_ch),
            np.array(image_green_ch),
            np.array(image_blue_ch)), -1)
        image = cv2.resize(image, (shape[0], shape[1]))
        return image

    @staticmethod
    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
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

def create_model(input_shape, n_out):
    input_tensor = Input(shape=(SIZE, SIZE, 3))
    #bn = BatchNormalization()(input_tensor)
    #conv = Conv2D(3,(3,3),padding='same',activation='relu')(bn)
    base_model = ResNet34(include_top=False,
                             weights='imagenet',
                             input_shape=(SIZE, SIZE, 3),input_tensor=input_tensor)

    x = base_model.output
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
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

epochs = [2,20]
batch_size = 16


# split data into train, valid

mskf = MultilabelStratifiedKFold(n_splits=5,shuffle=True,random_state=18)

y = np.zeros((len(train_dataset_info), 28))
for i in range(len(train_dataset_info)):
    y[i][train_dataset_info[i]['labels']] = 1
mskf.get_n_splits(train_dataset_info, y)
kf = mskf.split(train_dataset_info, y)
fold_id = 0
for train_indexes, valid_indexes in kf:

    checkpoint = ModelCheckpoint(MODEL_PATH + 'model_{}.h5'.format(fold_id), monitor='val_f1', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only=True)
    tensorboard = TensorBoard(MODEL_PATH + 'logs{}/'.format(fold_id))
    # reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
    #                                   verbose=1, mode='auto', epsilon=0.0001)
    # early = EarlyStopping(monitor="val_loss",
    #                      mode="min",
    #                      patience=6)
    callbacks_list = [checkpoint, tensorboard]

    # for checking if distribution is same
    #data['target_list'] = data['Target'].map(lambda x: [int(a) for a in x.split(' ')])
    #from itertools import chain
    #from collections import Counter
    #all_labels = list(chain.from_iterable(data['target_list'].values))
    #c_val = Counter(all_labels)
    #n_keys = c_val.keys()
    #data['target_vec'] = data['target_list'].map(lambda ck: [i in ck for i in range(len(data)+1)])
    #train_df = data.iloc[train_indexes]
    #valid_df = data.iloc[test_indexes]
    #print(train_df.shape[0], 'training masks')
    #print(valid_df.shape[0], 'validation masks')
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
    #train_sum_vec = np.sum(np.stack(train_df['target_vec'].values, 0), 0)
    #valid_sum_vec = np.sum(np.stack(valid_df['target_vec'].values, 0), 0)
    #ax1.bar(n_keys, [train_sum_vec[k] for k in n_keys])
    #ax1.set_title('Training Distribution')
    #ax2.bar(n_keys, [valid_sum_vec[k] for k in n_keys])
    #ax2.set_title('Validation Distribution')


    # create train and valid datagens
    aug = False
    if fold_id == 1:
        aug = True
    train_generator = data_generator.create_train(train_dataset_info[train_indexes],
                                                  batch_size, (SIZE, SIZE, 3), augument=aug)
    validation_generator = data_generator.create_train(train_dataset_info[valid_indexes],
                                                       batch_size, (SIZE, SIZE, 3), augument=False)

    # warm up model
    model = create_model(
        input_shape=(SIZE, SIZE, 4),
        n_out=28)

    for layer in model.layers:
        layer.trainable = False
    #model.layers[2].trainable = True
    model.layers[-1].trainable = True
    model.layers[-2].trainable = True
    model.layers[-3].trainable = True
    model.layers[-4].trainable = True
    model.layers[-5].trainable = True
    model.layers[-6].trainable = True

    from keras_losses import f1
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-03),
        metrics=['acc',f1])
    # model.summary()
    model.fit_generator(
        train_generator,
        steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
        validation_data=validation_generator,
        validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
        epochs=epochs[0],
        verbose=1)


    # train all layers
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['acc',f1])
    model.fit_generator(
        train_generator,
        steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
        validation_data=validation_generator,
        validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
        epochs=epochs[1],
        verbose=1,
        callbacks=callbacks_list)

    model.load_weights(MODEL_PATH + 'model_{}.h5'.format(fold_id))
    preds = np.zeros(shape=(len(valid_indexes),28))
    preds_05 = np.zeros(shape=(len(valid_indexes),28))
    y_true= np.zeros(shape=(len(valid_indexes),28))
    for i, info in tqdm(enumerate(train_dataset_info[valid_indexes])):
        image = data_generator.load_image(info['path'], (SIZE, SIZE, 3))
        score_predict = model.predict(image[np.newaxis])[0]
        thresh = max(score_predict[np.argsort(score_predict, axis=-1)[-5]], 0.2)
        preds[i][score_predict >= thresh] = 1
        preds_05[i][score_predict >= 0.5] = 1
        y_true[i][info['labels']]=1

    from sklearn.metrics import f1_score

    f1_res = f1_score(y_true, preds, average='macro')
    f1_res_05 = f1_score(y_true, preds_05, average='macro')
    print(f1_res)
    submit = pd.read_csv('assets/sample_submission.csv')
    predicted = []
    draw_predict = []

    for name in tqdm(submit['Id']):
        path = os.path.join('assets/test_256/', name)
        image = data_generator.load_image(path, (SIZE, SIZE, 3))
        score_predict = model.predict(image[np.newaxis])[0]
        draw_predict.append(score_predict)

        thresh = max(score_predict[np.argsort(score_predict, axis=-1)[-5]],0.2)
        label_predict = np.arange(28)[score_predict >= thresh]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    submit['Predicted'] = predicted
    #np.save('draw_predict_InceptionV3.npy', score_predict)
    submit.to_csv(MODEL_PATH + 'debug_submission_{}_{:.4}.csv'.format(fold_id,f1_res), index=False)

    fold_id += 1

