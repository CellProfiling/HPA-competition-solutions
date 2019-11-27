import os, sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

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

suffix = '_bce2'

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

normal_aug = A.Compose([A.Rotate((0,30),p=0.75),
                        A.RandomRotate90(p=1),
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightness(0.05),
                        A.RandomContrast(0.05),
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
                                              batch_size, (SIZE, SIZE, 3), augument=True, oversample_factor=0)
validation_generator = data_generator.create_train(train_dataset_info[valid_indexes],
                                                   1, (SIZE, SIZE, 3), augument=False,oversample_factor=0)

checkpoint = ModelCheckpoint(MODEL_PATH + 'model_f{}{}_f1_all.h5'.format(fold_id,suffix), monitor='val_f1_all', verbose=1,
                             save_best_only=True, mode='max', save_weights_only=True)
checkpoint2 = ModelCheckpoint(MODEL_PATH + 'model_f{}{}_bce.h5'.format(fold_id,suffix), monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)
tensorboard = TensorBoard(MODEL_PATH + 'logs{}{}'.format(fold_id,suffix) + '/')

f1_metric = F1Metric(validation_generator,len(valid_indexes)//1,1,28)
callbacks_list = [f1_metric, checkpoint, checkpoint2, tensorboard]


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


model.save_weights(MODEL_PATH + 'model_f{}_e{}{}.h5'.format(fold_id,epochs,suffix))

model.load_weights(MODEL_PATH + 'model_f{}{}_f1_all.h5'.format(fold_id,suffix))

preds = np.zeros(shape=(len(valid_indexes),28))
y_true= np.zeros(shape=(len(valid_indexes),28))


tta = 16
for i, info in tqdm(enumerate(train_dataset_info[valid_indexes])):
    image = data_generator.load_image(info['path'])
    images = np.zeros((16,256,256,3))
    for t in range(tta):
        images[t] = preprocess_input(data_generator.augment(normal_aug,image))
    score_predict = model.predict(images, batch_size=tta)
    score_predict = np.mean(score_predict,axis=0)
    preds[i] = score_predict
    y_true[i][info['labels']]=1

from sklearn.metrics import f1_score

#individual_f1_scores = np.zeros(28)
#for i in range(28):
#    individual_f1_scores[i] = f1_score(y_true[:,i],preds[:,i])
#individual_f1_scores = pd.DataFrame(individual_f1_scores,columns=['f1'])
#individual_f1_scores.to_csv(MODEL_PATH + f'summary_f1_score_f{fold_id}.csv',index=False)



#f1_res = f1_score(y_true, preds, average='macro')
f1_res_05 = f1_score(y_true, preds > 0.5, average='macro')

eps = 0.004
desired = {
0 : 0.36239782,
1 : 0.043841336,
2 : 0.075268817,
3 : 0.059322034,
4 : 0.075268817,
5 : 0.075268817,
6 : 0.043841336,
7 : 0.075268817,
8 : eps,
9 : eps,
10 : eps,
11 : 0.043841336,
12 : 0.043841336,
13 : 0.014198783,
14 : 0.043841336,
15 : eps,
16 : 0.028806584,
17 : 0.014198783,
18 : 0.028806584,
19 : 0.059322034,
20 : eps,
21 : 0.126126126,
22 : 0.028806584,
23 : 0.075268817,
24 : eps,
25 : 0.222493888,
26 : 0.028806584,
27 : eps
}

thresholds = np.linspace(0, 1, 101)
scores = np.array([f1_score(y_true, (preds > t).astype(int), average='macro') for t in thresholds])

threshold_best_index = np.argmax(scores)
score_best = scores[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
print('')
print('f1_best',score_best)
print('threshold_best',threshold_best)
print('')

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

plt.plot(thresholds, scores)
plt.plot(threshold_best, score_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("F1")
plt.title("Threshold vs F1 ({}, {})".format(threshold_best, score_best))
plt.legend()
plt.show()
plt.gcf().clear()

from sklearn.metrics import accuracy_score

acc = np.mean([accuracy_score(y_true[:,i],preds[:,i]>threshold_best) for i in range(28)])



SUBMISSION = False
if SUBMISSION:

    submit = pd.read_csv('Christof/assets/sample_submission.csv')
    predicted = []
    draw_predict = []
    test_preds = np.zeros((len(submit['Id']),28))
    for name in tqdm(submit['Id']):
        path = os.path.join('Christof/assets/test_rgb_256/', name)
        image = data_generator.load_image(path)
        images = np.zeros((16, 256, 256, 3))
        for t in range(tta):
            images[t] = preprocess_input(data_generator.augment(normal_aug, image))
        score_predict = model.predict(images, batch_size=tta)
        score_predict = np.mean(score_predict, axis=0)
        #draw_predict.append(score_predict)


        #label_predict = np.arange(28)[score_predict >= thresh]
        label_predict = np.arange(28)[score_predict >= threshold_best]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    submit['Predicted'] = predicted
    #np.save('draw_predict_InceptionV3.npy', score_predict)
    submit.to_csv(MODEL_PATH + 'submission_f{}{}_{:.4}.csv'.format(fold_id,suffix,score_best), index=False)

