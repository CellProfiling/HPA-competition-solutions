# Model description [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A CNN model that effectively uses large resolution image and trained with data distillation.

The model accept 1024*1024 input images and uses ImageNet pre-trained model for faster convergence and better generalizability.
In the training, we use gradient accumulation to overcome the small batch size due to large input image.
Multi-Label Stratification was used to split the dataset and up-sample the rare classes to alleviate the imbalance.
The training is done by two phases: freeze the backbone and only train the fully-connected layer then fine-tune.

With simple 5 fold average, post-processing and test time augmentation, this CNN model gets 0.553 on the test set.
