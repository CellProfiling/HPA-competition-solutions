# bestfitting [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Model description

A CNN model that effectively uses large resolution image and trained with data distillation.

The model accept 1024*1024 input images and uses ImageNet pre-trained model for faster convergence and better generalizability.
In the training, we use gradient accumulation to overcome the small batch size due to large input image.
Multi-Label Stratification was used to split the dataset and up-sample the rare classes to alleviate the imbalance.
The training is done by two phases: freeze the backbone and only train the fully-connected layer then fine-tune.

With simple 5 fold average, post-processing and test time augmentation, this CNN model gets 0.553 on the test set.

## Model usage

1. The basic Runtime Environment is python3.6, pytorch0.4.1, you can refer to requriements.txt to set up your environment.

2. Data process
cd src/data_process

2.1 Download v18 external data
python s1_download_hpa_v18.py

2.2 Resize tif image to 768 and 1536
python s2_resize_tif_image.py --dataset train --size 768
python s2_resize_tif_image.py --dataset test --size 768
python s2_resize_tif_image.py --dataset train --size 1536
python s2_resize_tif_image.py --dataset test --size 1536

2.3 Resize v18 external image to 512, 768 and 1536
python s3_resize_external_image.py --size 512
python s3_resize_external_image.py --size 768
python s3_resize_external_image.py --size 1536

2.4 Generate meta data
python s4_generate_meta.py

2.5 Search matching samples from training set and v18 external data
python s5_train_match_external.py

2.6 Search matching samples from test set
python s6_test_match_test.py

2.7 Split training set and validation set
python s7_generate_split.py

2.8 Calculate mean and std of images
python s8_generate_images_mean_std.py

2.9 Split training set and validation set for arcface models
python s9_generate_antibody_split.py

2.10 Modify wrong targets base on xml file from kaggle forum
python s10_generate_correct_external_meta.py

2.11 Generate leak test set
python s11_generate_test_leak_meta.py

3.Training
cd src/run

3.1 classification model
python train.py --out_dir external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds --gpu_id 0,1,2,3 --arch class_densenet121_dropout --scheduler Adam55 --epochs 55 --img_size 768 --crop_size 512 --batch_size 48 --split_name random_ext_folds5 --fold 0
python train.py --out_dir external_crop1024_focal_slov_hardlog_clean_class_densenet121_large_dropout_i1536_aug2_5folds --gpu_id 0,1,2,3 --arch class_densenet121_large_dropout --scheduler adam45 --epochs 45 --img_size 1536 --crop_size 1024 --batch_size 36 --split_name random_ext_noleak_clean_folds5 --fold 0
python train.py --out_dir external_crop512_focal_slov_hardlog_class_inceptionv3_dropout_i768_aug2_5folds --gpu_id 0,1,2,3 --arch class_inceptionv3_dropout --scheduler adam45 --epochs 45 --img_size 768 --crop_size 512 --batch_size 64 --split_name random_ext_noleak_clean_folds5 --fold 0
python train.py --out_dir external_crop1024_focal_slov_hardlog_clean_class_resnet34_dropout_i1536_aug2_5folds --gpu_id 0,1,2,3 --arch class_resnet34_dropout --scheduler adam45 --epochs 45 --img_size 1536 --crop_size 1024 --batch_size 48 --split_name random_ext_noleak_clean_folds5 --fold 0

3.2 metric learning model
python train_ml.py --out_dir face_all_class_resnet50_dropout_i768_aug2_5folds --gpu_id 0,1,2,3 --arch class_resnet50_dropout --scheduler FaceAdam --epochs 50 --img_size 768 --batch_size 32

4.Predicting
cd src/run
4.1 classification model
python test.py --out_dir external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds --gpu_id 0 --arch class_densenet121_dropout --img_size 768 --crop_size 512 --seeds 0,1,2,3 --batch_size 12 --fold 0 --augment default,flipud,fliplr,transpose,flipud_lr,flipud_transpose,fliplr_transpose,flipud_lr_transpose
python test.py --out_dir external_crop1024_focal_slov_hardlog_clean_class_densenet121_large_dropout_i1536_aug2_5folds --gpu_id 0 --arch class_densenet121_large_dropout --img_size 1536 --crop_size 1024 --seeds 0,1,2,3 --batch_size 8 --fold 0 --augment default,flipud,fliplr,transpose,flipud_lr,flipud_transpose,fliplr_transpose,flipud_lr_transpose
python test.py --out_dir external_crop512_focal_slov_hardlog_class_inceptionv3_dropout_i768_aug2_5folds --gpu_id 0 --arch class_inceptionv3_dropout --img_size 768 --crop_size 512 --seeds 0,1,2,3 --batch_size 24 --fold 0 --augment default,flipud,fliplr,transpose,flipud_lr,flipud_transpose,fliplr_transpose,flipud_lr_transpose
python test.py --out_dir external_crop1024_focal_slov_hardlog_clean_class_resnet34_dropout_i1536_aug2_5folds --gpu_id 0 --arch class_resnet34_dropout --img_size 1536 --crop_size 1024 --seeds 0,1,2,3 --batch_size 12 --fold 0 --augment default,flipud,fliplr,transpose,flipud_lr,flipud_transpose,fliplr_transpose,flipud_lr_transpose

4.2 metric learning model
python test_ml.py --out_dir face_all_class_resnet50_dropout_i768_aug2_5folds --gpu_id 0,1,2,3 --arch class_resnet50_dropout --img_size 768 --batch_size 32 --dataset test --predict_epoch 45

5.ensemble
cd src/ensemble

python ensemble_augment.py --fold 0 --epoch_name final --model_name external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds --augments default,flipud,fliplr,transpose,flipud_lr,flipud_transpose,fliplr_transpose,flipud_lr_transpose --do_valid 0 --do_test 1 --update 1 --seeds 0,1,2,3 --ensemble_type maximum
python ensemble_folds.py --en_cfgs external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds --do_valid 1 --do_test 1 --update 1

6.Post processing
cd src/post_processing

6.1 Search the most similar samples by metric learning model
python s1_calculate_distance.py --model_name face_all_class_resnet50_dropout_i768_aug2_5folds --epoch_name 045 --do_valid 0 --do_test 1

6.2 Modify submissions
python s2_modify_result.py --model_name external_crop1024_focal_slov_hardlog_clean_class_densenet121_large_dropout_i1536_aug2_5folds --face_model_name face_all_class_resnet50_dropout_i768_aug2_5folds --out_name d121_i1536_aug2_maximum_5folds_f012_max_test_ratio2_face_r50_i768 --threshold 0.65
