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
import albumentations as A
import warnings
warnings.filterwarnings("ignore")
from classification_models.resnet.models import ResNet34

MODEL_PATH = 'Christof/models/train_vs_test/ResNet34/1/'

SIZE = 256

# Load dataset info
path_to_train = 'Christof/assets/train_rgb_256/'
path_to_test = 'Christof/assets/test_rgb_256/'

train_data = pd.read_csv('Christof/assets/train.csv')
test_data = pd.read_csv('Christof/assets/sample_submission.csv')


normal_aug = A.Compose([A.Flip(p=0.75),
                        A.Rotate((-180,180))])

train_dataset_info = []
for name in train_data['Id']:
    train_dataset_info.append({
        'path': os.path.join(path_to_train, name),
        'labels': np.array(0)})
train_dataset_info = np.array(train_dataset_info)

test_dataset_info = []
for name in test_data['Id']:
    test_dataset_info.append({
        'path': os.path.join(path_to_test, name),
        'labels': np.array(1)})
test_dataset_info = np.array(test_dataset_info)

dataset_info = np.concatenate([train_dataset_info,test_dataset_info])
import random
random.shuffle(dataset_info,random=random.seed(23))

#rare_dataset_info = np.array([item for item in train_dataset_info if np.isin(item['labels'], rare_classes).any()])
#train_dataset_info = rare_dataset_info


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
                batch_labels = np.zeros((len(X_train_batch)))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(X_train_batch[i]['path'], shape)

                    if augument:
                        image = data_generator.augment(normal_aug,image)

                    batch_images.append(image)
                    batch_labels[i] = X_train_batch[i]['labels']
                yield np.array(batch_images, np.float32), batch_labels

    @staticmethod
    def load_image(path, shape):
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
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
import keras
from keras.layers import Cropping2D, UpSampling2D, Concatenate, MaxPooling1D, Maximum, Lambda, Add
from keras.models import Model

def create_model(input_shape, n_out):
    input_tensor = Input(shape=(SIZE, SIZE, 3))

    base_model = ResNet34(include_top=False,
                             weights='imagenet',
                             input_shape=(SIZE, SIZE, 3),input_tensor=input_tensor)


    x = GlobalMaxPooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
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
from keras_metrics import f1, f1_02
from keras_losses import  f1_loss
epochs = 100
batch_size = 16


# split data into train, valid
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf_gen = kf.split(dataset_info)

fold_id = 3
train_indexes, valid_indexes = next(kf_gen)

train_generator = data_generator.create_train(dataset_info[train_indexes],
                                              batch_size, (SIZE, SIZE, 3), augument=True)
validation_generator = data_generator.create_train(dataset_info[valid_indexes],
                                                   1, (SIZE, SIZE, 3), augument=False)

checkpoint = ModelCheckpoint(MODEL_PATH + 'model_{}.h5'.format(fold_id), monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)
tensorboard = TensorBoard(MODEL_PATH + 'logs{}'.format(fold_id) + '{}'.format(fold_id)  + '/')
# reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
#                                   verbose=1, mode='auto', epsilon=0.0001)
# early = EarlyStopping(monitor="val_loss",
#                      mode="min",
#                      patience=6)
callbacks_list = [checkpoint, tensorboard]


# warm up model
model = create_model(
    input_shape=(SIZE, SIZE, 3),
    n_out=1)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['acc'])
model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
    epochs=epochs,
    verbose=1,
    callbacks=callbacks_list)
