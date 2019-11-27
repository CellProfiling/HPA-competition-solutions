import os, sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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





model = create_model(
    input_shape=(SIZE, SIZE, 3),
    n_out=28)



model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-03),
    metrics=['acc'])



fold_id = 0

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

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path': os.path.join(path_to_train, name),
        'labels': np.array([int(label) for label in labels]),
        'fold':folds[name]})
train_dataset_info = np.array(train_dataset_info)

model.load_weights(MODEL_PATH + 'base_100/' + 'model_f{}.h5'.format(fold_id))

train_indexes, valid_indexes = get_fold_ids(fold_id, train_dataset_info)
#train_indexes, valid_indexes = next(kf)

validation_generator = data_generator.create_train(train_dataset_info[valid_indexes],
                                                   1, (SIZE, SIZE, 3), augument=False,oversample_factor=0)

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

f1_res = f1_score(y_true, preds, average='macro')
f1_res_05 = f1_score(y_true, preds_05, average='macro')
print(f1_res)
print(f1_res_05)

sample_submit = pd.read_csv('Christof/assets/sample_submission.csv')
test_path = 'Christof/assets/test_rgb_256/'
#images = np.zeros(shape=(len(sample_submit['Id']),SIZE,SIZE,3))
#augmented_images = np.zeros((len(sample_submit['Id']), SIZE, SIZE, 3))

preds = np.zeros(shape=(len(sample_submit['Id']),28))

batch_size = 64
num_batches = len(sample_submit['Id'])//batch_size +1



for batch_id in tqdm(range(num_batches)):
    start = batch_id * batch_size
    end = (batch_id+1) * batch_size
    test_batch = sample_submit['Id'][start:end]

    images = np.zeros((len(test_batch),SIZE,SIZE,3))
    #augmented_images = np.zeros((len(test_batch),SIZE,SIZE,3))
    for i, name in enumerate(test_batch):
        path = os.path.join('Christof/assets/test_rgb_256/', name)
        images[i] = data_generator.load_image(path)

    preds[start:end] = model.predict(images,batch_size=batch_size)


# add num classes in [1,5]

binary_preds = np.round(preds).astype(bool)

class_preds = [np.arange(28)[binary_pred] for binary_pred in binary_preds]
str_preds = [' '.join(str(l) for l in class_pred) for class_pred in class_preds]

sample_submit['Predicted'] = str_preds

np.save('best_ckpt_fold0_preds.npy', preds)
sample_submit.to_csv(MODEL_PATH + 'predictions/' + 'sub_best_ckpt_until_e100_fold0.csv', index=False)

"""
submit = pd.read_csv('Christof/assets/sample_submission.csv')
predicted = []
draw_predict = []

for name in tqdm(submit['Id']):
    path = os.path.join('Christof/assets/test_rgb_256/', name)
    image = data_generator.load_image(path)
    score_predict = model.predict(image[np.newaxis])[0]
    draw_predict.append(score_predict)

    #thresh = max(score_predict[np.argsort(score_predict, axis=-1)[-5]], 0.2)
    label_predict = np.arange(28)[score_predict >= 0.5]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
# np.save('draw_predict_InceptionV3.npy', score_predict)
submit.to_csv(MODEL_PATH + 'sec_submission.csv', index=False)
"""