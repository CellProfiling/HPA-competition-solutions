# NTU_MiRA [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Model description

A CNN model that effectively uses large resolution image and trained with data distillation.

The model accept 1024*1024 input images and uses ImageNet pre-trained model for faster convergence and better generalizability.
In the training, we use gradient accumulation to overcome the small batch size due to large input image.
Multi-Label Stratification was used to split the dataset and up-sample the rare classes to alleviate the imbalance.
The training is done by two phases: freeze the backbone and only train the fully-connected layer then fine-tune.

With simple 5 fold average, post-processing and test time augmentation, this CNN model gets 0.553 on the test set.

## Model source

The original model source can be found [here](https://github.com/CellProfiling/HPA-competition-solutions/tree/master/ntu_mira).

## Trained model files

The trained model files can be found [here](https://kth.box.com/s/5toiz1vbhmu1b7vrd10zthnruwaeqa4a)

## Dependencies

+ Python 3; TensorFlow >= 1.4.0
+ pip install albumentations
+ pip install iterative-stratification
+ pip install msgpack==0.5.6
+ Tensorpack@0.8.5 (https://github.com/tensorpack/tensorpack) (pip install -U git+https://github.com/ppwwyyxx/tensorpack.git@0.8.5)
+ OpenCV
+ Pre-trained [ResNet model](https://goo.gl/6XjK9V) from tensorpack model zoo.
