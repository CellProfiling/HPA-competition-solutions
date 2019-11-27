python3 submit_agn.py \
/home/t-zhga/protein-kaggle/input \
../model_weights/res18_256  \
/checkpoint/epoch_142_loss_0.4630_cv_0.6798_model.pth


python3 submit_agn.py \
/home/t-zhga/protein-kaggle/input \
../model_weights/res18_384  \
/checkpoint/epoch_147_loss_0.4510_cv_0.6880_model.pth


python3 submit_agn.py \
/home/t-zhga/protein-kaggle/input \
../model_weights/res18_512  \
/checkpoint/epoch_106_loss_0.5043_cv_0.6382_model.pth

python3 submit_avg.py 
