#!/usr/bin/env bash

#[inceptionv4]
python test.py --gpuid 0 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_1_epoch_12.pth.tar --batch_size 10
python test.py --gpuid 4 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_2_epoch_15.pth.tar --batch_size 10
python test.py --gpuid 5 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_3_epoch_12.pth.tar --batch_size 10
python test.py --gpuid 7 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_5_epoch_12.pth.tar --batch_size 10
python test.py --gpuid 2 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_8_epoch_22.pth.tar --batch_size 10
python test.py --gpuid 4 -imsize 800 -m inceptionv4_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv4_fc_flod_9_epoch_13.pth.tar --batch_size 10

#[inceptionv3]
python test.py --gpuid 5 -imsize 800 -m inceptionv3_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p inceptionv3_fc_flod_5_epoch_13.pth.tar --batch_size 10

#[xception]
python test.py --gpuid 7 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_0_epoch_14.pth.tar --batch_size 10
python test.py --gpuid 2 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_6_epoch_13.pth.tar --batch_size 10
python test.py --gpuid 4 -imsize 800 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_800_pretrain_5fold -p xception_fc_flod_7_epoch_13.pth.tar --batch_size 10

python test.py --gpuid 5 -imsize 512 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_5fold -p xception_fc_flod_0_epoch_17.pth.tar --batch_size 10
python test.py --gpuid 7 -imsize 512 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm_5fold -p xception_fc_flod_1_epoch_12.pth.tar --batch_size 10
python test.py --gpuid 7 -imsize 512 -m xception_fc -t offi_hpa_lr0.05_weightedsamper_mlsm -p xception_fc_flod_0_epoch_22.pth.tar --batch_size 10
python test.py --gpuid 5 -imsize 512 -m xception_fc -t offi_lr0.001_weightedsamper_mlsm_rms_lrexp_pretrain -p xception_fc_flod_0_epoch_6.pth.tar --batch_size 10

# get submission
python result_summary_tiny.py