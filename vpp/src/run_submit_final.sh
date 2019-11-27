#!/usr/bin/env bash

#[inceptionv3]
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain -p inceptionv3_fc_flod_0_epoch_8.pth.tar --batch_size 10 &
nohup python test.py --gpuid 1 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv3_fc_flod_0_epoch_18.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 2 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv3_fc_flod_1_epoch_12.pth.tar --batch_size 10 &
nohup python test.py --gpuid 2 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv3_fc_flod_1_epoch_20.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv3_fc_flod_2_epoch_15.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv3_fc_flod_2_epoch_21.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv3_fc_flod_3_epoch_21.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv3_fc_flod_4_epoch_14.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv3_fc_flod_5_epoch_13.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv3_fc_flod_6_epoch_12.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv3_fc_flod_7_epoch_12.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 1 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv3_fc_flod_8_epoch_12.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 2 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv3_fc_flod_9_epoch_12.pth.tar --batch_size 10 --with_feature &

nohup python test.py --gpuid 0 -imsize 650 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold -p inceptionv3_fc_flod_0_epoch_13.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 650 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_5fold_pretrain -p inceptionv3_fc_flod_3_epoch_13.pth.tar --batch_size 10 -bs 50 &


nohup python test.py -imsize 800 -m inceptionv3_fc -g 0 -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain -p inceptionv3_fc_flod_3_epoch_28.pth.tar -bs 50 &
nohup python test.py -imsize 800 -m inceptionv3_fc -g 1 -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain -p inceptionv3_fc_flod_3_epoch_9.pth.tar -bs 50 &
nohup python test.py -imsize 800 -m inceptionv3_fc -g 2 -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_1 -p inceptionv3_fc_flod_3_epoch_12.pth.tar -ci 1 -bs 50 &
nohup python test.py -imsize 800 -m inceptionv3_fc -g 3 -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_2 -p inceptionv3_fc_flod_3_epoch_12.pth.tar -ci 2  -bs 50 &
nohup python test.py -imsize 800 -m inceptionv3_fc -g 4 -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_3 -p inceptionv3_fc_flod_3_epoch_12.pth.tar -ci 3 -bs 50 &
nohup python test.py -imsize 800 -m inceptionv3_fc -g 5 -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_4 -p inceptionv3_fc_flod_3_epoch_12.pth.tar -ci 4 -bs 50 &
nohup python test.py -imsize 800 -m inceptionv3_fc -g 6 -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain -p inceptionv3_fc_flod_3_epoch_23.pth.tar  -bs 50 &
nohup python test.py -imsize 650 -m inceptionv3_fc -g 7 -t offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain -p inceptionv3_fc_flod_3_epoch_13.pth.tar  -bs 50 &


#[inceptionv4]
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_0_epoch_13.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_1_epoch_12.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_2_epoch_12.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_2_epoch_15.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 1 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_3_epoch_12.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 2 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_3_epoch_16.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_4_epoch_12.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_4_epoch_15.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_5_epoch_12.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_6_epoch_16.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_7_epoch_14.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_7_epoch_20.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 1 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_8_epoch_12.pth.tar --batch_size 10 &
nohup python test.py --gpuid 2 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_8_epoch_22.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_9_epoch_13.pth.tar --batch_size 10 --with_feature &

nohup python test.py --gpuid 0 -imsize 650 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold -p inceptionv4_fc_flod_0_epoch_17.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 650 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold -p inceptionv4_fc_flod_1_epoch_14.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 650 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold -p inceptionv4_fc_flod_2_epoch_13.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 650 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold -p inceptionv4_fc_flod_3_epoch_16.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 650 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold -p inceptionv4_fc_flod_4_epoch_15.pth.tar --batch_size 10 &
nohup python test.py --gpuid 1 -imsize 650 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold -p inceptionv4_fc_flod_5_epoch_13.pth.tar --batch_size 10 &
nohup python test.py --gpuid 2 -imsize 650 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold -p inceptionv4_fc_flod_6_epoch_13.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 650 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold -p inceptionv4_fc_flod_7_epoch_14.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 650 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold -p inceptionv4_fc_flod_8_epoch_16.pth.tar --batch_size 10 &


#[xception]
nohup python test.py --gpuid 0 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_0_epoch_14.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_1_epoch_18.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_2_epoch_18.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_3_epoch_18.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_3_epoch_12.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 1 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_4_epoch_14.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 2 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_5_epoch_12.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 2 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_5_epoch_17.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_6_epoch_13.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_7_epoch_13.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_8_epoch_13.pth.tar --batch_size 10 --with_feature &
nohup python test.py --gpuid 0 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_9_epoch_12.pth.tar --batch_size 10 --with_feature &

nohup python test.py --gpuid 0 -imsize 650 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_650_pretrain_10fold -p xception_fc_flod_0_epoch_8.pth.tar --batch_size 10 &


nohup python test.py --gpuid 0 -imsize 512 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_5fold -p xception_fc_flod_0_epoch_17.pth.tar --batch_size 10 &
nohup python test.py --gpuid 1 -imsize 512 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_5fold -p xception_fc_flod_1_epoch_12.pth.tar --batch_size 10 &
nohup python test.py --gpuid 2 -imsize 512 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_5fold -p xception_fc_flod_3_epoch_14.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 512 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm -p xception_fc_flod_0_epoch_22.pth.tar --batch_size 10 &
nohup python test.py --gpuid 0 -imsize 512 -m xception_fc -t offi_lr0.001_weightedsamper_mlsm_rms_lrexp_pretrain -p xception_fc_flod_0_epoch_6.pth.tar --batch_size 10 &

cd utils
python concat_h5_any_h5.py
cd ..
python test_mlp.py -t lr0.5_exp_bce_sgd_2layer_10fold -g 0
python test_mlp.py -t lr0.5_step_0.5_10_mlsm_sgd_2layer_10fold -g 0

# get submission
nohup python result_summary_final.py