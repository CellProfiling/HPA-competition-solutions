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
#from classification_models.resnet.models import ResNet101
from keras.applications.xception import Xception, preprocess_input

MODEL_PATH = 'Christof/models/Xception/1/'
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
from keras.layers import  GlobalAveragePooling2D, Concatenate
import keras
from keras.models import Model



def encoder(backbone):

    c1 = backbone.get_layer('block2_sepconv2_bn').output
    #c1 = ZeroPadding2D(((2,1),(2,1)),name='zero_pad_c1')(c1)
    c2 = backbone.get_layer('block3_sepconv2_bn').output
    #c2 = ZeroPadding2D(((1, 0), (1, 0)), name= 'zero_pad_c2')(c2)
    c3 = backbone.get_layer('block4_sepconv2_bn').output
    #c3 = ZeroPadding2D(((0, 1), (0, 1)), name='zero_pad_c3')(c3)
    c4 = backbone.get_layer('block13_sepconv2_bn').output
    #c5 = backbone.get_layer('block14_sepconv2_act').output

    short_cuts = [c1,c2,c3,c4]
    enc_out = backbone.output  # 8



    #return inp, c5, short_cuts
    return enc_out, short_cuts

def create_model(input_shape, n_out):
    input_tensor = Input(shape=(SIZE, SIZE, 3))

    base_model = Xception(include_top=False,
                             weights='imagenet',
                             input_shape=(SIZE, SIZE, 3),input_tensor=input_tensor)

    enc_out, short_cuts = encoder(base_model)
    x0 = GlobalAveragePooling2D()(enc_out)
    x1 = GlobalAveragePooling2D()(short_cuts[0])
    x2 = GlobalAveragePooling2D()(short_cuts[1])
    x3 = GlobalAveragePooling2D()(short_cuts[2])
    x4 = GlobalAveragePooling2D()(short_cuts[3])
    x = Concatenate()([x0,x1,x2,x3,x4])
    #x = Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    #x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
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
from keras_callbacks import F1Metric
from keras_metrics import f1, f1_02
from keras_losses import  f1_loss
epochs = [2,100]
batch_size = 16


# split data into train, valid

mskf = MultilabelStratifiedKFold(n_splits=5,shuffle=True,random_state=18)

#y = np.zeros((len(train_dataset_info), 28))
#for i in range(len(train_dataset_info)):
#    y[i][train_dataset_info[i]['labels']] = 1
#mskf.get_n_splits(train_dataset_info, y)
#kf = mskf.split(train_dataset_info, y)

train_indexes, valid_indexes = get_fold_ids(fold_id, train_dataset_info)
#train_indexes, valid_indexes = next(kf)

train_generator = data_generator.create_train(train_dataset_info[train_indexes],
                                              batch_size, (SIZE, SIZE, 3), augument=True, oversample_factor=3)
validation_generator = data_generator.create_train(train_dataset_info[valid_indexes],
                                                   1, (SIZE, SIZE, 3), augument=False,oversample_factor=0)

checkpoint = ModelCheckpoint(MODEL_PATH + 'model_f{}.h5'.format(fold_id), monitor='val_f1_all', verbose=1,
                             save_best_only=True, mode='max', save_weights_only=True)
tensorboard = TensorBoard(MODEL_PATH + 'logs{}_'.format(fold_id) + '/')

f1_metric = F1Metric(validation_generator,len(valid_indexes)//1,1,28)
callbacks_list = [f1_metric, checkpoint, tensorboard]


# warm up model
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
    epochs=epochs[1],
    verbose=1,
    callbacks=callbacks_list)
model.save_weights(MODEL_PATH + 'model_f{}_e{}.h5'.format(fold_id,epochs[1]))
model.load_weights(MODEL_PATH + 'model_f{}.h5'.format(fold_id))
preds = np.zeros(shape=(len(valid_indexes),28))
preds_05 = np.zeros(shape=(len(valid_indexes),28))
y_true= np.zeros(shape=(len(valid_indexes),28))
for i, info in tqdm(enumerate(train_dataset_info[valid_indexes])):
    image = data_generator.load_image(info['path'])

    score_predict = model.predict(image[np.newaxis])[0]
    thresh = max(score_predict[np.argsort(score_predict, axis=-1)[-5]], 0.2)
    preds[i][score_predict >= thresh] = 1
    preds_05[i][score_predict >= 0.5] = 1
    y_true[i][info['labels']]=1

from sklearn.metrics import f1_score

individual_f1_scores = np.zeros(28)
for i in range(28):
    individual_f1_scores[i] = f1_score(y_true[:,i],preds[:,i])
individual_f1_scores = pd.DataFrame(individual_f1_scores,columns=['f1'])
individual_f1_scores.to_csv(MODEL_PATH + f'summary_f1_score_f{fold_id}.csv',index=False)



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

        thresh = max(score_predict[np.argsort(score_predict, axis=-1)[-5]],0.2)
        label_predict = np.arange(28)[score_predict >= thresh]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    submit['Predicted'] = predicted
    #np.save('draw_predict_InceptionV3.npy', score_predict)
    submit.to_csv(MODEL_PATH + 'submission_f{}_{:.4}.csv'.format(fold_id,f1_res), index=False)

