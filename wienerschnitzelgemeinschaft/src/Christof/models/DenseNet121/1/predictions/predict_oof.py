import os, sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn.utils import shuffle
from ml_stratifiers import MultilabelStratifiedKFold
import albumentations as A
import warnings
warnings.filterwarnings("ignore")
from keras.applications.densenet import DenseNet121, preprocess_input

MODEL_PATH = 'Christof/models/DenseNet121/1/'
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
from sklearn.metrics import f1_score


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

    return model




batch_size = 32




model = create_model(
    input_shape=(SIZE, SIZE, 3),
    n_out=28)



model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-03),
    metrics=['acc'])

import scipy.optimize as opt

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

def F1_soft(preds,targs,th=0.5,d=50.0):
    preds = sigmoid_np(d*(preds - th))
    targs = targs.astype(np.float)
    score = 2.0*(preds*targs).sum(axis=0)/((preds+targs).sum(axis=0) + 1e-6)
    return score

def fit_val(x,y):
    params = 0.5*np.ones(28)
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x,y,p) - 1.0,
                                      wd*(p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p





folds = [0,1]

individual_f1_scores = np.zeros((len(folds),28))
thresholds = np.zeros((len(folds),28))
f1_th_scores = np.zeros((len(folds)))
f1_05_scores = np.zeros((len(folds)))
# split data into train, valid

augs = [[A.NoOp()],
        [A.HorizontalFlip(p=1.0)],
        [A.VerticalFlip(p=1.0)],
        #[A.RandomRotate90(p=1.0)],
        [A.HorizontalFlip(p=1.0),A.VerticalFlip(p=1.0)]]

for fold_id in folds:
    train_indexes, valid_indexes = get_fold_ids(fold_id, train_dataset_info)

    model.load_weights(MODEL_PATH + 'snaps/' + 'model_f{}.h5'.format(fold_id))

    y_true= np.zeros(shape=(len(valid_indexes),28))
    preds = np.zeros(shape=(len(valid_indexes),28))
    images = np.zeros((len(train_dataset_info[valid_indexes]),SIZE,SIZE,3))
    augmented_images = np.zeros((len(train_dataset_info[valid_indexes]),SIZE,SIZE,3))
    for i, info in tqdm(enumerate(train_dataset_info[valid_indexes])):
        images[i] = data_generator.load_image(info['path'])
        y_true[i][info['labels']]=1

    for aug in augs:
        for i,img in enumerate(images):
            augmented_image = img
            for transform in aug:
                augmented_image = transform.apply(augmented_image)
            augmented_images[i] = augmented_image
        preds += model.predict(augmented_images,batch_size=64,verbose=True)
    preds /= len(augs)

    print('optimizing th for fold {}'.format(fold_id))
    th = fit_val(preds,y_true)
    thresholds[fold_id] = th
    print('Thresholds: ',th)

    f1_05 = f1_score(y_true, preds>0.5, average='macro')
    f1_th = f1_score(y_true, preds>th, average='macro')

    print('F1 macro: ',f1_th)
    print('F1 macro (th = 0.5): ',f1_05)

    f1_th_scores[fold_id] = f1_th
    f1_05_scores[fold_id] = f1_05

    for j in range(28):
        individual_f1_scores[fold_id,j] = f1_score(y_true[:,j],preds[:,j]>0.5)



individual_f1_scores = pd.DataFrame(individual_f1_scores,columns=['f1'])
individual_f1_scores.to_csv(MODEL_PATH + f'summary_f1_score_f{fold_id}.csv',index=False)

