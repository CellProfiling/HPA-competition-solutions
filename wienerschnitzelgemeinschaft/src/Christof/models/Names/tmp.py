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

MODEL_PATH = 'Christof/models/ResNet34/tests/22/'
exp_suffix = '0'

SIZE = 256

# Load dataset info
path_to_train = 'Christof/assets/train_rgb_256/'
data = pd.read_csv('Christof/assets/train.csv').sample(frac=0.2)

data['text0'] = data['Id'].apply(lambda x: x.split('-')[0])
data['text1'] = data['Id'].apply(lambda x: x.split('-')[1])
data['text2'] = data['Id'].apply(lambda x: x.split('-')[2])
data['text3'] = data['Id'].apply(lambda x: x.split('-')[3])
data['text4'] = data['Id'].apply(lambda x: x.split('-')[4])

def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

cat_cols = ['text1','text3']

encoders = [{} for cat in cat_cols]

from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(data.index.values,test_size=0.25, random_state=23)

for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(data.loc[train_indices, cat].astype(str).unique())}
    data[cat] = data[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    print('Done')

embed_sizes = [len(encoder) + 1 for encoder in encoders]



from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras.losses import binary_crossentropy, mse

categorical_inputs = []
for cat in cat_cols:
    categorical_inputs.append(Input(shape=[1], name=cat))

categorical_embeddings = []
for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
#categorical_logits = Flatten()(categorical_embeddings[0])
categorical_logits = Dense(32,activation='relu')(categorical_logits)


out = Dense(28, activation='sigmoid')(categorical_logits)

model = Model(inputs = categorical_inputs, outputs=out)
from keras_metrics import f1
model.compile(optimizer='adam',loss=binary_crossentropy, metrics=[f1])

X_train = [data['text1'][train_indices].values,data['text3'][train_indices].values]
X_val = [data['text1'][val_indices].values,data['text3'][val_indices].values]

labels = data['Target'].str.split(' ')

labels_train = labels[train_indices]
y_train = np.zeros((len(labels_train),28))
for i in range(len(labels_train)):
    y_train[i][[int(label) for label in labels_train.iloc[i]]] = 1

labels_val = labels[val_indices]
y_val = np.zeros((len(labels_val),28))
for i in range(len(labels_val)):
    y_val[i][[int(label) for label in labels_val.iloc[i]]] = 1



model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=50, batch_size=128,)

