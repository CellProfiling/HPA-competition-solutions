# bestfitting [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Model description

A CNN multi-label classification model to predict labels of each sample.

Multi-Label Stratification was used to split the dataset into a training set and a validation set. The performance of the model was estimated using focal loss over the validation set. A Densenet121 acts as the backbone of the model. The GlobalMaxPool and GlobalAvgPool layers of the final CNN feature map were concatenated before being fed to two fully connected layers to calculate the probability of each class.

Image augmentation was performed by flipping the image, rotating it 90 degrees, and randomly cropping out 1024 ⨉ 1024 pixel (px) patches. To improve the model’s predictive power, multiple random crops were taken, and the maximum probability among them was calculated when predicting on the test set.

A combined loss function of focal loss, Lovasz loss, and hard example log loss was used for training the model. It was optimized using an Adam optimizer with a step learning rate of [30, 15, 7.5, 3, 1] ⨉ 1e-5 for [25, 5, 5, 5, 5] epochs respectively. The output was thresholded using the ratio of labels in the training set.

Using this method, the CNN reached 0.565 Macro F1 on the test set when averaging predictions from 5 folds.

## Model source

The original model source can be found [here](https://github.com/CellProfiling/HPA-competition-solutions/tree/master/bestfitting).

## Trained model files

The trained model files can be found [here](https://kth.box.com/s/gw43cvngx6quknq8ana9um1xx3ajhi4a).

## Model usage

1. The basic Runtime Environment is python3.6, pytorch0.4.1, you can refer to requriements.txt to set up your environment.

2. Data process

    1. Go to subdirectory

        ```sh
        cd src/data_process
        ```

    2. Download v18 external data

        ```sh
        python s1_download_hpa_v18.py
        ```

    3. Resize tif image to 768 and 1536

        ```sh
        python s2_resize_tif_image.py --dataset train --size 768
        python s2_resize_tif_image.py --dataset test --size 768
        python s2_resize_tif_image.py --dataset train --size 1536
        python s2_resize_tif_image.py --dataset test --size 1536
        ```

    4. Resize v18 external image to 512, 768 and 1536

        ```sh
        python s3_resize_external_image.py --size 512
        python s3_resize_external_image.py --size 768
        python s3_resize_external_image.py --size 1536
        ```

    5. Generate meta data

        ```sh
        python s4_generate_meta.py
        ```

    6. Search matching samples from training set and v18 external data

        ```sh
        python s5_train_match_external.py
        ```

    7. Search matching samples from test set

        ```sh
        python s6_test_match_test.py
        ```

    8. Split training set and validation set

        ```sh
        python s7_generate_split.py
        ```

    9. Calculate mean and std of images

        ```sh
        python s8_generate_images_mean_std.py
        ```

    10. Split training set and validation set for arcface models

        ```sh
        python s9_generate_antibody_split.py
        ```

    11. Modify wrong targets base on xml file from kaggle forum

        ```sh
        python s10_generate_correct_external_meta.py
        ```

    12. Generate leak test set

        ```sh
        python s11_generate_test_leak_meta.py
        ```

3. Training

    1. Go to subdirectory

        ```sh
        cd src/run
        ```

    2. classification model

        ```sh
        python train.py \
            --out_dir external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds \
            --gpu_id 0,1,2,3 --arch class_densenet121_dropout --scheduler Adam55 --epochs 55 \
            --img_size 768 --crop_size 512 --batch_size 48 --split_name random_ext_folds5 --fold 0
        ```

        ```sh
        python train.py \
            --out_dir external_crop1024_focal_slov_hardlog_clean_class_densenet121_large_dropout_i1536_aug2_5folds \
            --gpu_id 0,1,2,3 --arch class_densenet121_large_dropout --scheduler adam45 --epochs 45 \
            --img_size 1536 --crop_size 1024 --batch_size 36 --split_name random_ext_noleak_clean_folds5 --fold 0
        ```

        ```sh
        python train.py \
            --out_dir external_crop512_focal_slov_hardlog_class_inceptionv3_dropout_i768_aug2_5folds \
            --gpu_id 0,1,2,3 --arch class_inceptionv3_dropout --scheduler adam45 --epochs 45 \
            --img_size 768 --crop_size 512 --batch_size 64 --split_name random_ext_noleak_clean_folds5 --fold 0
        ```

        ```sh
        python train.py \
            --out_dir external_crop1024_focal_slov_hardlog_clean_class_resnet34_dropout_i1536_aug2_5folds \
            --gpu_id 0,1,2,3 --arch class_resnet34_dropout --scheduler adam45 --epochs 45 \
            --img_size 1536 --crop_size 1024 --batch_size 48 --split_name random_ext_noleak_clean_folds5 --fold 0
        ```

    3. metric learning model

        ```sh
        python train_ml.py \
            --out_dir face_all_class_resnet50_dropout_i768_aug2_5folds \
            --gpu_id 0,1,2,3 --arch class_resnet50_dropout --scheduler FaceAdam --epochs 50 \
            --img_size 768 --batch_size 32
        ```

4. Predicting

    1. Go to subdirectory

        ```sh
        cd src/run
        ```

    2. classification model

        ```sh
        python test.py \
            --out_dir external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds \
            --gpu_id 0 --arch class_densenet121_dropout \
            --img_size 768 --crop_size 512 --seeds 0,1,2,3 --batch_size 12 --fold 0 \
            --augment default,flipud,fliplr,transpose,flipud_lr,flipud_transpose,fliplr_transpose,flipud_lr_transpose
        ```

        ```sh
        python test.py \
            --out_dir external_crop1024_focal_slov_hardlog_clean_class_densenet121_large_dropout_i1536_aug2_5folds \
            --gpu_id 0 --arch class_densenet121_large_dropout \
            --img_size 1536 --crop_size 1024 --seeds 0,1,2,3 --batch_size 8 --fold 0 \
            --augment default,flipud,fliplr,transpose,flipud_lr,flipud_transpose,fliplr_transpose,flipud_lr_transpose
        ```

        ```sh
        python test.py \
            --out_dir external_crop512_focal_slov_hardlog_class_inceptionv3_dropout_i768_aug2_5folds \
            --gpu_id 0 --arch class_inceptionv3_dropout \
            --img_size 768 --crop_size 512 --seeds 0,1,2,3 --batch_size 24 --fold 0 \
            --augment default,flipud,fliplr,transpose,flipud_lr,flipud_transpose,fliplr_transpose,flipud_lr_transpose
        ```

        ```sh
        python test.py \
            --out_dir external_crop1024_focal_slov_hardlog_clean_class_resnet34_dropout_i1536_aug2_5folds \
            --gpu_id 0 --arch class_resnet34_dropout \
            --img_size 1536 --crop_size 1024 --seeds 0,1,2,3 --batch_size 12 --fold 0 \
            --augment default,flipud,fliplr,transpose,flipud_lr,flipud_transpose,fliplr_transpose,flipud_lr_transpose
        ```

    3. metric learning model

        ```sh
        python test_ml.py \
            --out_dir face_all_class_resnet50_dropout_i768_aug2_5folds \
            --gpu_id 0,1,2,3 --arch class_resnet50_dropout \
            --img_size 768 --batch_size 32 --dataset test --predict_epoch 45
        ```

5. Ensemble

    1. Go to subdirectory

        ```sh
        cd src/ensemble
        ```

    2. Make ensemble

        ```sh
        python ensemble_augment.py \
            --fold 0 --epoch_name final \
            --model_name external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds \
            --augments default,flipud,fliplr,transpose,flipud_lr,flipud_transpose,fliplr_transpose,flipud_lr_transpose \
            --do_valid 0 --do_test 1 --update 1 --seeds 0,1,2,3 --ensemble_type maximum
        ```

        ```sh
        python ensemble_folds.py \
            --en_cfgs external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds \
            --do_valid 1 --do_test 1 --update 1
        ```

6. Post processing

    1. Go to subdirectory

        ```sh
        cd src/post_processing
        ```

    2. Search the most similar samples by metric learning model

        ```sh
        python s1_calculate_distance.py \
            --model_name face_all_class_resnet50_dropout_i768_aug2_5folds --epoch_name 045 \
            --do_valid 0 --do_test 1
        ```

    3. Modify submissions

        ```sh
        python s2_modify_result.py \
            --model_name external_crop1024_focal_slov_hardlog_clean_class_densenet121_large_dropout_i1536_aug2_5folds \
            --face_model_name face_all_class_resnet50_dropout_i768_aug2_5folds \
            --out_name d121_i1536_aug2_maximum_5folds_f012_max_test_ratio2_face_r50_i768 --threshold 0.65
        ```
