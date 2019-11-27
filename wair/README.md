# WAIR

## Model description

The solution is mainly based on CNN models. Seven models derived from densenet121 (3 variants) [1], densenet169 [1], ibn_densenet121 [2], se-resnextnet50 [3]and xception [4] were trained and evaluated on image data provided by Kaggle and Human Protein Atlas (HPA) websites.

To train each model, Adamax [5] was used as optimizer and a cycle learning rate [6] with warm up was set to schedule. Independent models were finally ensembled through averaging the predicated probability for each class from each model.

The public tools involved are pytorch, opencv and scikit-learn. It takes 20-30 hours to train each model, and the full training process will take a week, using four Nvidia GTX 1080Ti GPUs.

## Model usage

### CONTENTS

- src                          : contains code for training and predicting
- comp_mdl                     : model binaries used in generating solution
- SETTINGS.json                : specifies the path to the train, test, model, and output directories.
- Dockerfile                   : specifies the exact version of all of the packages used
- data                         : data directory, contains raw data, downloaded data, processed data, corrected labels and cross validation dataset.
- download_external_data.py    : code to download external data from HPA wetsite
- prepare_data.py              : code to proprecess kaggle data and external data
- calibrate_model.py           : code to reproduce the best score on leader board using precomputed neural network predictions
- predict.py                   : code to predict and genarate the submission with trained model
- train_model.py               : code to retrain models

### HARDWARE

The following specs were used to create the original solution

- Ubuntu 16.04.4 LTS
- 40 vCPUs, 64 GB memory
- 4 x NVIDIA 1080Ti

### SOFTWARE

Python packages are detailed separately in `Dockerfile`.

- Python 3.6.7
- CUDA 8.0.61
- cudnn 7.1.2
- nvidia drivers 410.78
- Dockerfile : It specifies the exact version of all of the packages used.

### DATA SETUP

#### Download and extract train.zip and test.zip to data/raw directory

```sh
kaggle competitions download -c human-protein-atlas-image-classification -f train.zip
kaggle competitions download -c human-protein-atlas-image-classification -f test.zip
mkdir -p data/raw
unzip train.zip -d data/raw/train
unzip test.zip -d data/raw/test
```

#### Download external data and save images to data/external_data/jpgs directory

```sh
python3 download_external_data.py
```

- Download data to EXTERNAL_DATA_DIR.

#### Kaggle data and external date were preprocessed and converted to RGBY images (saved to data/processed)

```sh
python3 prepare_data.py
```

- Read training data from RAW_DATA_DIR and EXTERNAL_DATA_DIR
- Preprocess and convert to RGBY images.
- Save the cleaned data to ALL_IMAGE_TRAIN_DATA_CLEAN_DIR and IMAGE_TEST_DATA_CLEAN_DIR

#### Corrected labels

The labels are stored in data directory. The labels of kaggle images are in `train_process.csv`. The labels of kaggle images and external images are in `all_more_train_process.csv`.

#### Cross validation dataset

There are two versions of 5-fold cross validation dataset. There are six models trained using 5-fold_V1 and there is one model trained using 5-fold_V2. 5-fold_V1 was produced by the random kfold algorithm provided by scikit-learn package. 5-fold_V2 was produced by the [MultilabelStratifiedKFold algorithm](https://github.com/trent-b/iterative-stratification). The latter performed better. If there was more time the 5-fold_v2 would only have been used. The validation sets are saved in `data/processed` directory.

### Inferencing

1. Very fast prediction

    - Precomputed neural network predictions (TRAINED_MODEL_CHECKPOINT_DIR + '/ensemble') are used to ensemble and generate the submission which could reach the same score (0.571 on private lb) as the best score on leader board during the competition.

        ```sh
        python3 calibrate_model.py
        ```

    - Read precomputed predictions from TRAINED_MODEL_CHECKPOINT_DIR.
    - Ensemble.
    - Save predictions to SUBMISSION_DIR.

2. Ordinary prediction

    - Predict with trained model, and genarate the submission after predicting.

        ```sh
        python3 /predict.py
        ```

    - Read test images from IMAGE_TEST_DATA_CLEAN_DIR.
    - Load models from TRAINED_MODEL_CHECKPOINT_DIR.
    - Save predictions to SUBMISSION_DIR.

### Training

- Train the model.

    ```sh
    python3 train_model.py
    ```

- Read training data from ALL_IMAGE_TRAIN_DATA_CLEAN_DIR and training label from ALL_LABEL_TRAIN_DATA_CLEAN_PATH
- Train 7 models.
- Save your model to MODEL_CHECKPOINT_DIR.

## Reference

1. Huang G, Liu Z, Van Der Maaten L, et al. Densely connected convolutional networks[C]//2017
IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2017: 2261-2269.
2. Pan X, Luo P, Shi J, et al. Two at once: Enhancing learning and generalization capacities via
ibn-net[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 464-479.
3. Hu J, Shen L, Sun G. Squeeze-and-excitation networks[J]. arXiv preprint arXiv:1709.01507,
2017, 7.
4. Chollet F. Xception: Deep learning with depthwise separable convolutions[J]. arXiv preprint,
2017: 1610.02357.
5. Kingma D P, Ba J. Adam: A method for stochastic optimization[J]. arXiv preprint
arXiv:1412.6980, 2014.
6. Smith L N. Cyclical learning rates for training neural networks[C]//Applications of Computer
Vision (WACV), 2017 IEEE Winter Conference on. IEEE, 2017: 464-472.
