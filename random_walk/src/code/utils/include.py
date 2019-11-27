import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import data_parallel
from torch.utils.data.sampler import *
from torchvision import transforms
import random

import os
import gc
import cv2
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from scipy import ndimage
from timeit import default_timer as timer
from sklearn.metrics import f1_score
from PIL import Image
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
# from skmultilearn.model_selection import iterative_train_test_split

#INPUT_DIR = '../input'
INPUT_DIR = '/home/t-zhga/protein-kaggle/input'
label_map_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',
14:  'Microtubules',
15:  'Microtubule ends',
16:  'Cytokinetic bridge',
17:  'Mitotic spindle',
18:  'Microtubule organizing center',
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',
22:  'Cell junctions',
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',
27:  'Rods & rings' }

n_splits = 1
test_size = 0.1
random_state = 42
