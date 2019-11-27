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
import albumentations as A
import warnings
warnings.filterwarnings("ignore")
#from classification_models.resnet.models import ResNet101
from keras.applications.densenet import DenseNet121, preprocess_input

MODEL_PATH = 'Christof/models/DenseNet121/1/'
SIZE = 256
fold_ids = [1,2]


# Load dataset info
path_to_train = 'Christof/assets/train_rgb_256/'
data = pd.read_csv('Christof/assets/train.csv')

import pickle
with open('Russ/folds2.pkl', 'rb') as f:
    folds = pickle.load(f)

data['fold'] = data['Id'].apply(lambda x: folds[x])

def get_fold_ids(fold_id,data_set_info, shuff = True):
    fold_info = np.array([item['fold'] for item in data_set_info])
    val_ids = np.where(fold_info == fold_id)[0]
    train_ids = np.where(fold_info != fold_id)[0]
    if shuff:
        shuffle(val_ids)
        shuffle(train_ids)
    return train_ids, val_ids

normal_aug = A.Compose([A.OneOf([A.Rotate((-180,180)),
                                 A.Rotate((-180,180),border_mode=cv2.BORDER_CONSTANT)]),
                        A.Flip(p=0.75)
                        ])

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path': os.path.join(path_to_train, name),
        'labels': np.array([int(label) for label in labels]),
        'fold':folds[name]})
train_dataset_info = np.array(train_dataset_info)

counts = np.zeros(28)
for item in train_dataset_info:
    for l in item['labels']:
        counts[l] = counts[l] + 1

counts = counts / len(train_dataset_info)
rare_classes = np.where(counts < 0.005)


from classification_models.resnet import preprocess_input
class data_generator:

    @staticmethod
    def create_train(dataset_info, batch_size, shape, augument=True, oversample_factor = 0):
        assert shape[2] == 3

        if oversample_factor > 0:

            rare_dataset_info = np.array([item for item in dataset_info if np.isin(item['labels'], rare_classes).any()])
            for i in range(oversample_factor):
                dataset_info = np.append(dataset_info,rare_dataset_info)
        while True:
            dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), 28))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(X_train_batch[i]['path'])
                    #rare = np.isin(X_train_batch[i]['labels'], rare_classes).any()

                    if augument:
                        image = data_generator.augment(normal_aug,image)

                    batch_images.append(image)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    @staticmethod
    def load_image(path):
        image = cv2.imread(path + '.png', cv2.IMREAD_UNCHANGED)
        image = preprocess_input(image)
        return image

    @staticmethod
    def augment(aug,image):
        image_aug = aug(image=image)['image']
        return image_aug


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
import keras
from keras.models import Model

def create_model(input_shape, n_out):
    input_tensor = Input(shape=(SIZE, SIZE, 3))

    base_model = DenseNet121(include_top=False,
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
from keras_callbacks import F1Metric, SnapshotCallbackBuilder
from keras_metrics import f1, f1_02
from keras_losses import  f1_loss
epochs = [100,120]
batch_size = 32


# split data into train, valid

mskf = MultilabelStratifiedKFold(n_splits=5,shuffle=True,random_state=18)

#y = np.zeros((len(train_dataset_info), 28))
#for i in range(len(train_dataset_info)):
#    y[i][train_dataset_info[i]['labels']] = 1
#mskf.get_n_splits(train_dataset_info, y)
#kf = mskf.split(train_dataset_info, y)

for fold_id in fold_ids:
    train_indexes, valid_indexes = get_fold_ids(fold_id, train_dataset_info)
    #train_indexes, valid_indexes = next(kf)

    train_generator = data_generator.create_train(train_dataset_info[train_indexes],
                                                  batch_size, (SIZE, SIZE, 3), augument=True, oversample_factor=3)
    validation_generator = data_generator.create_train(train_dataset_info[valid_indexes],
                                                       1, (SIZE, SIZE, 3), augument=False,oversample_factor=0)

    checkpoint = ModelCheckpoint(MODEL_PATH + 'base_100/' + 'model_f{}.h5'.format(fold_id), monitor='val_f1_all', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only=True)
    tensorboard = TensorBoard(MODEL_PATH + 'base_100/' + 'logs{}_'.format(fold_id) + '/')
    f1_metric = F1Metric(validation_generator,len(valid_indexes)//1,1,28)
    callbacks_list = [f1_metric, checkpoint, tensorboard]


    model = create_model(
        input_shape=(SIZE, SIZE, 3),
        n_out=28)


    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['acc',f1,f1_02])

    model.fit_generator(
        train_generator,
        steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
        validation_data=validation_generator,
        validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
        epochs=epochs[0],
        verbose=1,
        callbacks=callbacks_list)
    model.save_weights(MODEL_PATH +'base_100/' + 'model_f{}_e{}.h5'.format(fold_id,epochs[0]))

    checkpoint = ModelCheckpoint(MODEL_PATH + 'snaps/' + 'model_f{}.h5'.format(fold_id), monitor='val_f1_all', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only=True)
    tensorboard = TensorBoard(MODEL_PATH +'snaps/' + 'logs{}_'.format(fold_id) + '/')
    ss = SnapshotCallbackBuilder(epochs[1], nb_snapshots=4, init_lr=0.0002, model_path=MODEL_PATH + 'snaps/',
                                 model_prefix=f'model_f{fold_id}').get_callbacks()
    f1_metric = F1Metric(validation_generator, len(valid_indexes) // 1, 1, 28)
    callbacks_list = [f1_metric, checkpoint, tensorboard] + ss



    model.load_weights(MODEL_PATH +'base_100/' + 'model_f{}_e{}.h5'.format(fold_id,epochs[0]))
    model.fit_generator(
        train_generator,
        steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
        validation_data=validation_generator,
        validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
        epochs=epochs[1],
        verbose=1,
        callbacks=callbacks_list)
    model.save_weights(MODEL_PATH + 'snaps/'+ 'model_f{}_e{}.h5'.format(fold_id,epochs[1] + 100))
