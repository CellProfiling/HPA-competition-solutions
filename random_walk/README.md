# Random Walk [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Model description

An ensemble model of three different resolutions based on single attention gated network.

Resnet18 was used as backbone, and the feature maps of the last three blocks were used to generate soft-attention and self-gating. The self-attention mechanism generates a gating signal that is end-to-end trainable, which allows the network to contextualise local information useful for prediction. The attended features at last 3 blocks of Resnet18 are combined for a final prediction by aggregating the mean.

The original images (512 ⨉ 512) were cropped into 3 different sizes (256 ⨉ 256, 384 ⨉ 384, and 512 ⨉ 512) to fit 3 separate AGN models, before finally ensembling them. A combined loss function of soft f1 loss and focal loss was used for training the model.

The solution only used a single model and 512 ⨉ 512 PNG files with HPAv18 external data, without any use of cross validation.

## Model source

The original model source can be found [here](https://github.com/CellProfiling/HPA-competition-solutions/tree/master/random_walk).

## Trained model files

The trained model files can be found [here](https://kth.box.com/s/2lkwn52zs0glid0wro8riidxriax7pkq).

## Model usage

1. for single fold run with different resolution, just change and run code/train_agn.py transforms.RandomCrop(256) with (256,384,512)
2. for predict models and ensemble sub, just run code/submit.sh
