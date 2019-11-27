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

MODEL_PATH = 'Christof/models/pytorch/ResNet34/2/'


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
        #self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
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

        #x = self.avg_pool(x)

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




epochs = 5
batch_size = 32


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


#train_model(train_generator,validation_generator,model,criterion,optimizer)

mname = 'ResNet34'
fold = 0

print('')
print('*' * 50)
print(mname + ' fold ' + str(fold))
print('*' * 50)
bname = mname + '/' + 'best_' + str(fold) + '.pth'

# Network definition
net = Resnet(num_classes=28, pretrained=True)


gpu_id = 0
if gpu_id >= 0:
    print('Using GPU: {} '.format(gpu_id))
    torch.cuda.set_device(device=gpu_id)
    # net.cuda()
    device = "cuda"
    # device = "cpu"
    net.train()
    net.to(device)

# Logging into Tensorboard
# log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
from tensorboardX import  SummaryWriter

log_dir = MODEL_PATH + 'logs/' + '_' + str(fold)
writer = SummaryWriter(log_dir=log_dir)

from torch.optim.lr_scheduler import ReduceLROnPlateau
#     scheduler = LambdaLR(optimizer, lr_lambda=cyclic_lr)
#     scheduler.base_lrs = list(map(lambda group: 1.0, optimizer.param_groups))
#     scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=5, verbose=True,
#                                   threshold=0.0, threshold_mode='abs')

#scheduler = StepLR(optimizer, step_size=p['step_size'], gamma=p['gamma'])

torch.cuda.empty_cache()

len_train = len(train_dataset_info[train_indexes])
len_val = len(train_dataset_info[valid_indexes])
print('Training on ' + str(len_train) +
      ' and validating on ' + str(len_val))


#utils.generate_param_report(os.path.join(save_dir, mname + '.txt'), p)

# number of batches
num_batches_tr = len_train // batch_size
num_batches_val = len_val // batch_size

#print('Image size:', image_size)
#print('Batch size:', p['trainBatch'])
print('Number of batches per epoch: ', num_batches_tr)
# print('Learning rate: ', p['lr'])
print('')

running_loss_tr = 0.0
running_loss_ts = 0.0
aveGrad = 0
bname = MODEL_PATH + 'best_' + str(fold) + '.pth'
# print("Training Network")
history = {}
history['epoch'] = []
history['train'] = []
history['val'] = []
history['delta'] = []
history['f1'] = []
history['time'] = []
best_val = -999
bad_epochs = 0

import timeit

start_time = timeit.default_timer()
total_time = 0
prev_lr = 999
resume_epoch = 0
# Main Training and Testing Loop

from sklearn.metrics import  f1_score
for epoch in range(resume_epoch, epochs):

    #         if (epoch > 0) and (epoch % p['epoch_size'] == 0):
    #             lr_ = utils.lr_poly(p['lr'], epoch, nEpochs, 0.9)
    #             print('(poly lr policy) learning rate', lr_)
    #             print('')
    #             optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=p['momentum'],
    #                                   weight_decay=p['wd'])

    #scheduler.step()
    #lr = optimizer.param_groups[0]['lr']
    #if lr != prev_lr:
    #    print('learning rate = %.6f' % lr)
    #    prev_lr = lr

    net.train()

    train_loss = []

    for s in range(num_batches_tr):

        inputs, labels = next(train_generator)
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels).float()

        #             # load training batch
        #             time_cbatch = time.time()

        #             batch = bg_augmenter.get_batch()

        #             # need to transpose for pytorch model
        #             inputs = np.array(batch.images_aug).astype(np.float32).transpose(0, 3, 1, 2)

        #             gts = np.array(batch.targets).astype(int)

        #             inputs = torch.from_numpy(inputs).float()
        #             gts = torch.from_numpy(gts).int()

        #             time_cbatch = time.time() - time_cbatch

        inputs = inputs.type(torch.float).to(device)
        labes = labels.to(device)
        # gts = gts.type(torch.float).to(device)

        # predictions are on a logit scale
        logits = net(inputs)

        loss = criterion(logits, labels.to(device))

        optimizer.zero_grad()
        loss.backward()

        ## adamw
        #for group in optimizer.param_groups:
        #    for param in group['params']:
        #        param.data = param.data.add(-p['wd'] * group['lr'], param.data)

        optimizer.step()
        train_loss.append(loss.item())
        running_loss_tr += loss.item()

    # validation
    net.eval()

    with torch.no_grad():

        val_loss = []
        val_predictions = []
        val_targets = []

        for s in range(num_batches_val):
            inputs, labels = next(validation_generator)
            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels).float()
            # tta horizontal flip
            #inputs2 = inputs[:, :, :, ::-1].copy()
            #inputs2 = torch.from_numpy(inputs2)

            inputs = inputs.type(torch.float).to(device)
            #inputs2 = inputs2.type(torch.float).to(device)

            # predictions are on a logit scale
            logits = net(inputs)
            #logits2 = net(inputs2)
            #logits = (logits + logits2) / 2.0

            loss = criterion(logits, labels.to(device))

            running_loss_ts += loss.item()
            val_loss.append(loss.item())

            # save results to compute F1 on validation set
            preds = logits.cpu().detach().numpy()
            gt = labels.cpu().detach().numpy()

            val_predictions.append(preds)
            val_targets.append(gt)

        vps = np.vstack(val_predictions)
        vts = np.vstack(val_targets)

        # competition metric
        vf1 = f1_score(vts, (vps > 0).astype(int), average='macro')

        if vf1 > best_val:
            star = '*'
            best_val = vf1
            torch.save(net.state_dict(), bname)
            bad_epochs = 0
        else:
            star = ' '
            bad_epochs += 1

        # print progress
        # running_loss_ts = running_loss_ts / num_img_ts

        tl = np.mean(train_loss)
        vl = np.mean(val_loss)

        stop_time = timeit.default_timer()
        diff_time = stop_time - start_time
        total_time += diff_time / 60.
        start_time = timeit.default_timer()

        print('epoch %d  train %6.4f  val %6.4f  delta %6.4f  f1 %6.4f%s  time %2.0f%s\n' % \
              (epoch, tl, vl, vl - tl, vf1, star, diff_time, 's'))
        writer.add_scalar('loss', tl, epoch)
        writer.add_scalar('val_loss', vl, epoch)
        writer.add_scalar('delta', vl - tl, epoch)
        writer.add_scalar('val_f1', vf1, epoch)
        writer.add_scalar('time', diff_time, epoch)
        # print('Running Loss: %f\n' % running_loss_ts)
        # print('Mean Loss: %f\n' % np.mean(val_loss))
        running_loss_tr = 0
        running_loss_ts = 0

        history['epoch'].append(epoch)
        history['train'].append(tl)
        history['val'].append(vl)
        history['f1'].append(vf1)
        history['time'].append(diff_time)

        #if bad_epochs > p['patience']:
        #    print('early stopping, best validation loss %6.4f, total time %4.1f minutes \n' % \
        #          (best_val, total_time))
        #    break

writer.close()

# plot history
fig, (ax_loss) = plt.subplots(1, 1, figsize=(8, 4))
ax_loss.plot(history['epoch'], history['train'], label="Train loss")
ax_loss.plot(history['epoch'], history['val'], label="Validation loss")
plt.show()
plt.gcf().clear()