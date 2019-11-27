import os, sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"



import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from classification_models.resnet.models import ResNet18
import albumentations as A

import cv2
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

MODEL_PATH = 'Christof/models/GAPNet/13_ext_512crop/'
suffix = '_br'
SIZE = 512

sampling_weights = [   2.6473,   35.0588 ,   8.2069  , 19.3439 ,  16.0145 ,  13.3245 ,  32.8644,
   10.607 ,  551.3   ,  501.1818 , 787.5714 ,  25.8523 ,  39.0301,   51.644,
   30.0846 ,1470.1333 ,  62.8262,  190.1034 ,  39.3084 ,  23.2126 , 170.9457
,    8.2592,   33.2609 ,   9.6889  , 92.2678  ,  4.19 ,    99.3333 ,3150.2857]

sample_weights_ext = [   2.6728,   41.1617 ,  10.3068 ,  42.4172  , 22.9729 ,  21.9808  , 26.8267
,   11.5358 , 474.8659 , 486.7375 , 492.8987  , 66.963 ,   50.2763  , 82.7609,
   45.0683, 1854.2381,  100.3582 , 319.1721  , 76.5762  , 33.424 ,  272.3007,
    7.3664 ,  39.4319  , 10.239 ,  734.6981 ,   2.548 ,  196.6616 , 638.3443]


data = pd.read_csv('Christof/assets/train.csv')
path_to_train = 'Christof/assets/train_rgb_1024/'

val_aug = A.Compose([#A.Rotate((0,30),p=0.75),
                        A.RandomRotate90(p=1),
                        A.HorizontalFlip(p=0.5),
                        #A.RandomCrop(512,512),
                        #A.RandomBrightness(0.05),
                        #A.RandomContrast(0.05),
                        A.IAAAffine(translate_percent=10,rotate=45,shear=10, scale=(0.9,1.1)),
                        #A.RandomAffine(degrees=45, translate=(0.1,0.1), shear=10, scale=(0.9,1.1))
                        A.Normalize(mean=(0.08069, 0.05258, 0.05487), std=(0.1300, 0.0879, 0.1386),
                                    max_pixel_value=255.)
                        ])

class data_generator:

    @staticmethod
    def create_train(dataset_info, batch_size, shape, augument=None, weighted_sample = True):
        assert shape[2] == 3

        if weighted_sample:
            p = np.array([item['weight'] for item in dataset_info])
            p = p/np.sum(p)
        else:
            p = None

        while True:
            #dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                #end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = np.random.choice(dataset_info,batch_size,p=p)
                batch_labels = np.zeros((len(X_train_batch), 28))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(X_train_batch[i]['path'], shape)
                    #image = preprocess_input(image)
                    #rare = np.isin(X_train_batch[i]['labels'], rare_classes).any()

                    if augument:
                        image = data_generator.augment(augument,image)

                    batch_images.append(image)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    @staticmethod
    def load_image(path, shape):
        image = cv2.imread(path + '.png', cv2.IMREAD_UNCHANGED)
        image = image[512:,512:,:]
        return image

    @staticmethod
    def augment(aug,image):
        image_aug = aug(image=image)['image']
        return image_aug



from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, Concatenate, Input, Conv2D
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
import keras
from keras.models import Model


from keras.layers import Layer, InputSpec
from keras import initializers
from keras.constraints import Constraint
import keras.backend as K

from keras.layers import Reshape, Permute, multiply
def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def encoder(backbone):

    c0 = backbone.get_layer('relu0').output

    c1 = backbone.get_layer('stage2_unit1_relu1').get_output_at(0)  # 128
    c2 = backbone.get_layer('stage3_unit1_relu1').output  # 63
    c3 = backbone.get_layer('stage4_unit1_relu1').output  # 32
    enc_out = backbone.get_layer('relu1').output  # 16
    #enc_out = backbone.output  # 8

    short_cuts = [c0,c1,c2,c3]
    return enc_out, short_cuts

from keras.layers import BatchNormalization
def create_model(input_shape, n_out):
    input_tensor = Input(shape=(SIZE, SIZE, 3))
    #bn = BatchNormalization()(input_tensor)
    #conv = Conv2D(3,(3,3),padding='same',activation='relu')(bn)
    base_model = ResNet18(include_top=False,
                             weights='imagenet',
                             input_shape=(SIZE, SIZE, 3),input_tensor=input_tensor)

    enc_out, short_cuts = encoder(base_model)
    x0 = GlobalAveragePooling2D()(squeeze_excite_block(enc_out))
    x1 = GlobalAveragePooling2D()(squeeze_excite_block(short_cuts[0]))
    x2 = GlobalAveragePooling2D()(squeeze_excite_block(short_cuts[1]))
    x3 = GlobalAveragePooling2D()(squeeze_excite_block(short_cuts[2]))
    x4 = GlobalAveragePooling2D()(squeeze_excite_block(short_cuts[3]))
    x = Concatenate()([x0,x1,x2,x3,x4])
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    #x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)

    model = Model(input_tensor, output)

    # transfer imagenet weights
    #res_img = ResNet34(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
    #offset = 2
    #for i, l in enumerate(base_model.layers[offset+1:]):
    #    l.set_weights(res_img.layers[i + 1].get_weights())

    return model



#submit = pd.read_csv('Christof/assets/sample_submission.csv')
tta = 8



folds_csv = pd.read_csv('Christof/assets/stratified_5fold.csv',index_col='Id')

data['fold'] = data['Id'].apply(lambda x: folds_csv.loc[x])



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
        'fold':folds_csv.loc[name]})
train_dataset_info = np.array(train_dataset_info)




#fold_np = train_dataset_info[val_ids]
oof_preds = np.zeros((len(train_dataset_info),28))

model = create_model(
    input_shape=(SIZE, SIZE, 3),
    n_out=28)

for fold_id in range(5):

    train_ids, val_ids =get_fold_ids(fold_id,train_dataset_info,shuff=False)

    model.load_weights(MODEL_PATH + f'model_loss_{fold_id}.h5')
    for val_id in tqdm(val_ids):
        path = train_dataset_info[val_id]['path']
        image = data_generator.load_image(path, (SIZE, SIZE, 3))
        images = [data_generator.augment(val_aug, image) for _ in range(tta)]
        tta_predicts = model.predict(np.array(images))
        oof_preds[val_id] = np.median(tta_predicts,axis = 0)


for c in range(28):
    data[f'pred_{c}'] = oof_preds[:,c]

data.to_csv(MODEL_PATH + f'oof_pred{suffix}.csv', index = False)

ls = data['Target'].values
oofs = data[[f'pred_{c}' for c in range(28)]].values

def str2y(item):
    np_labels = np.zeros((28,))
    labels = item.split(' ')
    int_labels = [int(label) for label in labels]
    np_labels[int_labels] = 1
    return np_labels

ls2 = [str2y(item) for item in tqdm(ls)]
ls2 = np.array(ls2)


from sklearn.metrics import f1_score

f1_score(ls2,oofs>0.3, average='macro')