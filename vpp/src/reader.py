
import numpy as np
import torch
from config import *
import pathlib
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.preprocessing import MultiLabelBinarizer
from imgaug import augmenters as iaa
import random
# set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)


def weighted_random_sampler(train_data_list,config):
    # class_weight_logdenped = [1.0, 3.01, 1.95, 2.79, 2.61, 2.31, 3.23, 2.2, 6.17, 6.34, 6.81, 3.15, 3.61, 3.86, 3.17,
    #                           7.1, 3.87, 4.8, 3.34, 2.84, 4.99, 1.91, 3.46, 2.15, 4.37, 1.13, 4.35, 7.74]

    class_weight_logdenped = [
            1., 5.9675195, 2.89230479, 5.75118779, 4.6351863,
            4.26514326, 5.45811323, 3.19660961, 14.47703644, 14.84415252,
            15.14004386, 6.91607887, 6.859, 8.1202995, 6.32468877,
            19.23918481, 8.48476847, 11.92630068, 7.32001592, 5.48454474,
            11.99490825, 2.39485678, 6.29696919, 2.99981866, 12.06364688,
            1., 10.38509393, 16.49538699
        ]
    temp_train_data_list = train_data_list.copy()
    sample_weight = []
    for it_train_target in temp_train_data_list['Target']:
        _max_class_weight = 1
        for _it_class_target in np.array(map(int, it_train_target.split(' '))):
            if class_weight_logdenped[_it_class_target] > _max_class_weight:
                _max_class_weight = class_weight_logdenped[_it_class_target]
        sample_weight.append(_max_class_weight)
    return sample_weight

class HumanDataset(Dataset):
    def __init__(self, images_df, base_path, augument=True, mode="train", config=''):
        self.config = config
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)
        self.images_df = images_df.copy()
        self.augument = augument
        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / x)
        self.mlb = MultiLabelBinarizer(classes=np.arange(0, self.config.num_classes))
        self.mlb.fit(np.arange(0, self.config.num_classes))
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):
        if not isinstance(index, int):
            index = index.item()
        X = self.read_images(index)
        if not self.mode == "test":
            labels = np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))
            y = np.eye(self.config.num_classes, dtype=np.float)[labels].sum(axis=0)
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.augument:
            X = self.augumentor(X)
        X = T.Compose([T.ToPILImage(), T.ToTensor()])(X)
        return X.float(), y

    def read_images(self, index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        # use only rgb channels
        if filename.find('hpa_') >= 0:
            filename = self.config.train_data_hpa + os.path.basename(filename).replace('hpa_', '')

        if os.path.exists(filename + "_red.png"):
            r = np.array(Image.open(filename + "_red.png"))
            g = np.array(Image.open(filename + "_green.png"))
            b = np.array(Image.open(filename + "_blue.png"))
            y = np.array(Image.open(filename + "_yellow.png"))
        else:
            r = np.array(Image.open(filename + "_red.jpg"))
            g = np.array(Image.open(filename + "_green.jpg"))
            b = np.array(Image.open(filename + "_blue.jpg"))
            y = np.array(Image.open(filename + "_yellow.jpg"))

        if self.config.channels == 4:
            images = np.zeros(shape=(r.shape[0], r.shape[1], 4))
        else:
            images = np.zeros(shape=(r.shape[0], r.shape[1], 3))
        if self.config.channels == 4:
            images[:, :, 0] = r.astype(np.uint8)
            images[:, :, 1] = g.astype(np.uint8)
            images[:, :, 2] = b.astype(np.uint8)
            if self.config.channels == 4:
                images[:, :, 3] = y.astype(np.uint8)
        else:
            images[:, :, 0] = r.astype(np.uint8)
            images[:, :, 1] = y.astype(np.uint8)
            images[:, :, 2] = b.astype(np.uint8)
        images = images.astype(np.uint8)
        # images = np.stack(images,-1)
        if self.config.img_height == images.shape[0]:
            return images
        elif images.shape[0] == 1024:
            if self.config.cropindex == 1:
                return images[0:800, 0:800, :]
            elif self.config.cropindex == 2:
                return images[1024 - 800:1024, 0:800, :]
            elif self.config.cropindex == 3:
                return images[0:800, 1024 - 800:1024, :]
            elif self.config.cropindex == 4:
                return images[1024 - 800:1024, 1024 - 800:1024, :]
            else:
                return cv2.resize(images, (self.config.img_weight, self.config.img_height))
        else:
            return cv2.resize(images, (self.config.img_weight, self.config.img_height))

    def augumentor(self, image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),

            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

