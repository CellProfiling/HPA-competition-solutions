#!/bin/bash
python 01_tain_sub_res34.py &
wait
python 02_train_sub_res34_swa.py &
wait
python 03_train_sub_res50xt.py &
wait
python 04_train_sub_res101xt.py &
wait
python 05_train_sub_res101xt_swa.py &
wait
python 06_train_sub_wrn_swa.py &
wait
python 07_train_sub_wrn_extra.py &
wait
python 08_train_sub_res50xt_extra_4chn.py &
wait
python 10_train_sub_res34_extra_3chn.py &
wait
python 11_train_sub_res34_extra_4chn.py &
wait
python 12_train_sub_res101xt_extra_4chn.py &
wait
python 13_train_sub_res18_extra_4chn.py &
wait
python 14_train_sub_res18_extra_4chn_256.py &
wait
python 16_train_sub_res34_extra_4chn_256.py &

wait
python ens12a.py &
python ens103a.py &

wait
python sub_leakCor_12.py &
python sub_leakCor_103.py &

wait
python enstw36a.py &
python enstw39b1a.py