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



tta_augs = [[A.NoOp()],
        [A.HorizontalFlip(p=1.0)],
        [A.VerticalFlip(p=1.0)],
        #[A.RandomRotate90(p=1.0)],
        [A.HorizontalFlip(p=1.0),A.VerticalFlip(p=1.0)]]

folds = [0,1,2,3,4]

sample_submit = pd.read_csv('Christof/assets/sample_submission.csv')
test_path = 'Christof/assets/test_rgb_256/'
#images = np.zeros(shape=(len(sample_submit['Id']),SIZE,SIZE,3))
#augmented_images = np.zeros((len(sample_submit['Id']), SIZE, SIZE, 3))

preds = np.zeros(shape=(len(sample_submit['Id']),28))

batch_size = 64
num_batches = len(sample_submit['Id'])//batch_size +1

for fold_id in folds:
    model.load_weights(MODEL_PATH + 'snaps/' + 'model_f{}.h5'.format(fold_id))

    for batch_id in tqdm(range(num_batches)):
        start = batch_id * batch_size
        end = (batch_id+1) * batch_size
        test_batch = sample_submit['Id'][start:end]

        images = np.zeros((len(test_batch),SIZE,SIZE,3))
        augmented_images = np.zeros((len(test_batch),SIZE,SIZE,3))
        for i, name in enumerate(test_batch):
            path = os.path.join('Christof/assets/test_rgb_256/', name)
            images[i] = data_generator.load_image(path)

        for aug in tta_augs:
            for i, img in enumerate(images):

                augmented_image = img
                for transform in aug:
                    augmented_image = transform.apply(augmented_image)
                augmented_images[i] = augmented_image

            preds[start:end] += model.predict(augmented_images,batch_size=batch_size)

preds /= (len(folds) * len(tta_augs))

# add num classes in [1,5]

binary_preds = np.round(preds).astype(bool)

class_preds = [np.arange(28)[binary_pred] for binary_pred in binary_preds]
str_preds = [' '.join(str(l) for l in class_pred) for class_pred in class_preds]

sample_submit['Predicted'] = str_preds

np.save('preds.npy', preds)
sample_submit.to_csv(MODEL_PATH + 'predictions/' + 'sub_4tta_best_ckpt_folds0-4.csv', index=False)

