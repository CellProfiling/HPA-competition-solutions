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
from classification_models.resnet.models import ResNet34
from classification_models.resnet import preprocess_input
#from keras.applications.densenet import DenseNet121, preprocess_input

MODEL_PATH = 'Christof/models/ResNet34/9/'
SIZE = 256
fold_id = 0

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
                    image = preprocess_input(image)
                    batch_images.append(image)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    @staticmethod
    def load_image(path):
        image = cv2.imread(path + '.png', cv2.IMREAD_UNCHANGED)

        return image

    @staticmethod
    def augment(aug,image):
        image_aug = aug(image=image)['image']
        return image_aug


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from keras.layers import  GlobalAveragePooling2D, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
import keras
from keras.models import Model
from keras.initializers import he_uniform
def create_model(input_shape, n_out):
    input_tensor = Input(shape=(SIZE, SIZE, 3))

    base_model = ResNet34(include_top=False,
                             weights='imagenet',
                             input_shape=(SIZE, SIZE, 3),input_tensor=input_tensor)

    x = base_model.output
    x1 = GlobalMaxPooling2D()(x)
    x2 = GlobalAveragePooling2D()(x)
    x = Concatenate()([x1,x2])
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out,activation='sigmoid')(x)
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
from keras_losses import f1_loss
epochs = 100
batch_size = 32


# split data into train, valid

#mskf = MultilabelStratifiedKFold(n_splits=10,shuffle=True,random_state=18)

#y = np.zeros((len(train_dataset_info), 28))
#for i in range(len(train_dataset_info)):
#    y[i][train_dataset_info[i]['labels']] = 1
#mskf.get_n_splits(train_dataset_info, y)
#kf = mskf.split(train_dataset_info, y)

train_indexes, valid_indexes = get_fold_ids(fold_id, train_dataset_info)
#train_indexes, valid_indexes = next(kf)

train_generator = data_generator.create_train(train_dataset_info[train_indexes],
                                              batch_size, (SIZE, SIZE, 3), augument=False, oversample_factor=0)
validation_generator = data_generator.create_train(train_dataset_info[valid_indexes],
                                                   1, (SIZE, SIZE, 3), augument=False,oversample_factor=0)

checkpoint = ModelCheckpoint(MODEL_PATH + 'model_f{}_bce_noaug.h5'.format(fold_id), monitor='val_f1_all', verbose=1,
                             save_best_only=True, mode='max', save_weights_only=True)
tensorboard = TensorBoard(MODEL_PATH + 'logs{}_bce_noaug'.format(fold_id) + '/')

f1_metric = F1Metric(validation_generator,len(valid_indexes)//1,1,28)
callbacks_list = [f1_metric, checkpoint, tensorboard]


# warm up model
model = create_model(
    input_shape=(SIZE, SIZE, 3),
    n_out=28)


from keras_losses import KerasFocalLoss
from keras.losses import binary_crossentropy


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




def bce_f1(y_true,y_pred):#

    return binary_crossentropy(y_true,y_pred) + soft_f1_loss(y_true,y_pred)


model.compile(loss=binary_crossentropy,
              optimizer=Adam(lr=0.0001, clipnorm=1.0),metrics=['acc'])


model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
    epochs=100,
    verbose=1,
    callbacks= callbacks_list)


model.save_weights(MODEL_PATH + 'model_f{}_e{}_bce_noaug.h5'.format(fold_id,epochs))
model.load_weights(MODEL_PATH + 'model_f{}_e50.h5'.format(fold_id))
preds = np.zeros(shape=(len(valid_indexes),28))
preds_05 = np.zeros(shape=(len(valid_indexes),28))
y_true= np.zeros(shape=(len(valid_indexes),28))
for i, info in tqdm(enumerate(train_dataset_info[valid_indexes])):
    image = data_generator.load_image(info['path'])
    image = preprocess_input(image)
    score_predict = model.predict(image[np.newaxis])[0]
    thresh = 0.2
    preds[i][score_predict >= thresh] = 1
    preds_05[i][score_predict >= 0.5] = 1
    y_true[i][info['labels']]=1

from sklearn.metrics import f1_score

#individual_f1_scores = np.zeros(28)
#for i in range(28):
#    individual_f1_scores[i] = f1_score(y_true[:,i],preds[:,i])
#individual_f1_scores = pd.DataFrame(individual_f1_scores,columns=['f1'])
#individual_f1_scores.to_csv(MODEL_PATH + f'summary_f1_score_f{fold_id}.csv',index=False)



f1_res = f1_score(y_true, preds, average='macro')
f1_res_05 = f1_score(y_true, preds_05, average='macro')
print(f1_res)
print(f1_res_05)

SUBMISSION = False
if SUBMISSION:

    submit = pd.read_csv('Christof/assets/sample_submission.csv')
    predicted = []
    draw_predict = []

    for name in tqdm(submit['Id']):
        path = os.path.join('Christof/assets/test_rgb_256/', name)
        image = data_generator.load_image(path)
        score_predict = model.predict(image[np.newaxis])[0]
        draw_predict.append(score_predict)

        #thresh = max(score_predict[np.argsort(score_predict, axis=-1)[-5]],0.2)
        #label_predict = np.arange(28)[score_predict >= thresh]
        label_predict = np.arange(28)[score_predict >= 0.2]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    submit['Predicted'] = predicted
    #np.save('draw_predict_InceptionV3.npy', score_predict)
    submit.to_csv(MODEL_PATH + 'submission_f{}_{:.4}_02.csv'.format(fold_id,f1_res), index=False)

