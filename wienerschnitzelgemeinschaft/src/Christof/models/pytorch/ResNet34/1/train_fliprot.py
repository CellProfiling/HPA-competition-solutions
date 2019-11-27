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

MODEL_PATH = 'Christof/models/pytorch/ResNet34/1/'
exp_suffix = 'fliprotcrop'

SIZE = 256

# Load dataset info
path_to_train = 'Christof/assets/train_rgb_256/'
data = pd.read_csv('Christof/assets/train.csv')

normal_aug = A.Compose([A.OneOf([A.Rotate((-180,180)),
                                 A.Rotate((-180,180),border_mode=cv2.BORDER_CONSTANT)]),
                        A.RandomSizedCrop(min_max_height=(128, 256), height=SIZE, width=SIZE, p=0.5),
                        A.Flip(p=0.75)
                        ])

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path': os.path.join(path_to_train, name),
        'labels': np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)

counts = np.zeros(28)
for item in train_dataset_info:
    for l in item['labels']:
        counts[l] = counts[l] + 1

counts = counts / len(train_dataset_info)
rare_classes = np.where(counts < 0.005)

#rare_dataset_info = np.array([item for item in train_dataset_info if np.isin(item['labels'], rare_classes).any()])
#train_dataset_info = rare_dataset_info


from classification_models.resnet import preprocess_input
class data_generator:

    @staticmethod
    def create_train(dataset_info, batch_size, shape, augument=True, oversample_factor = 0):
        assert shape[2] == 3



        if oversample_factor > 0:

            rare_dataset_info = np.array([item for item in dataset_info if np.isin(item['labels'], rare_classes).any()])
            #rare_dataset_info = shuffle(rare_dataset_info)
            for i in range(oversample_factor):
            #dataset_info
                dataset_info = np.append(dataset_info,rare_dataset_info)
        while True:
            dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), 28))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(X_train_batch[i]['path'], shape)
                    #image = preprocess_input(image)
                    rare = np.isin(X_train_batch[i]['labels'], rare_classes).any()

                    if augument:
                        image = data_generator.augment(normal_aug,image)

                    batch_images.append(np.transpose(image,axes=(2,0,1)))
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    @staticmethod
    def load_image(path, shape):
        image = cv2.imread(path + '.png', cv2.IMREAD_UNCHANGED)
        return image

    @staticmethod
    def augment(aug,image):
        image_aug = aug(image=image)['image']
        return image_aug



from torchvision.models import resnet34
import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
import time
from torch.utils.data import DataLoader

#device = torch.device('cpu')
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class Resnet(nn.Module):

    def __init__(self, num_classes,pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        layers = list(resnet34(pretrained).children())[:-2]
        layers += [AdaptiveConcatPool2d()]
        self.encoder = nn.Sequential(*layers)

        self.csize = 1024 * 1 * 1
        self.do1 = nn.Dropout(p=0.5)
        self.lin1 = nn.Linear(1024, 256)
        self.act1 = nn.ReLU()
        self.do2 = nn.Dropout(0.5)
        self.lin2 = nn.Linear(256, num_classes)
        self.act2 = nn.Sigmoid()

    def forward(self, x):

        # set to True for debugging
        print_sizes = False
        if print_sizes:
            print('')
            print('x', x.shape)



        x = self.encoder(x)
        x = x.view(-1, self.csize)
        if print_sizes: print('view', x.size())

        x = self.do1(x)
        if print_sizes: print('do1', x.size())

        x = self.lin1(x)
        if print_sizes: print('lin1', x.size())
        x = self.act1(x)
        x = self.do2(x)
        x = self.lin2(x)
        if print_sizes: print('lin2', x.shape)
        x = self.act2(x)
        return x





model = Resnet(num_classes=28)
model.cuda()
# cut off last fc

#model2 = nn.Sequential(*list(model.children())[:-1])

#top = [nn.ReLU(),nn.Dropout(0.5),nn.Linear(256, 28),nn.Sigmoid()]
#,nn.ReLU(),nn.Dropout(0.5),nn.Linear(256, 28),nn.Sigmoid()
#model.children()
#model = model.to(device)

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(),lr = 0.0001)

#model = nn.Sequential(*(list(model.children()) + top))

def train_model(train_gen, val_gen, model, criterion, optimizer, num_epochs=10, use_gpu=True):
    train_steps = 1553
    val_steps = 50

    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = 10000.0

    for epoch in range(num_epochs):
        tic = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_batch = 0

            # Iterate over data.
            if phase == 'train':
                steps = train_steps
            else:
                steps =val_steps

            for s in range(steps):
                if phase == 'train':
                    inputs, labels = next(train_gen)
                else:
                    inputs, labels = next(val_gen)
                inputs = torch.from_numpy(inputs)
                labels = torch.from_numpy(labels)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda()).float()
                else:
                    inputs, labels = Variable(inputs), Variable(labels).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)


                #_, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                #running_corrects += torch.sum(preds == labels.data)
                running_batch += 1

            epoch_loss = running_loss / running_batch
            #epoch_acc = running_corrects / dataset_sizes[phase]

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            # deep copy the model
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
        toc = time.time()
        print(toc-tic)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model











epochs = [2,120]
batch_size = 16


# split data into train, valid

mskf = MultilabelStratifiedKFold(n_splits=5,shuffle=True,random_state=18)

y = np.zeros((len(train_dataset_info), 28))
for i in range(len(train_dataset_info)):
    y[i][train_dataset_info[i]['labels']] = 1
mskf.get_n_splits(train_dataset_info, y)
kf = mskf.split(train_dataset_info, y)
fold_id = 0
train_indexes, valid_indexes = next(kf)

train_generator = data_generator.create_train(train_dataset_info[train_indexes],
                                              batch_size, (SIZE, SIZE, 3), augument=True, oversample_factor=0)
validation_generator = data_generator.create_train(train_dataset_info[valid_indexes],
                                                   1, (SIZE, SIZE, 3), augument=False, oversample_factor=0)

train_model(train_generator,validation_generator,model,criterion,optimizer)

