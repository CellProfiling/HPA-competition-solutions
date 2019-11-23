# Model description

This is an ensemble of three models. The best performing single-model (achieving private LB score 0.543) uses the cropping window technique and Xception (referred to as CWXception). The other two CNN models are SE-ResNext50 and InceptionV3, respectively.

In the first model (CWXception), we first resized the original image into 960×960, then cropped ten windows based on green channel signals with 0.4x the size of the resized image which thus yielded ten 384×384 windowed images and fed to a pre-trained Xception CNN. To verify that the windows capture the highly localized subcomponents, we built an interactive [GUI tool](https://i.imgur.com/OJ4NA9r.gifv) to superimpose the prediction scores and the locations onto the original image. Additionally, the cropping window provides another way of ensemble/augmentation thus improves the robustness of model predictions.

Both SE-ResNext50 and InceptionV3 models are trained on the whole 512x512 images. In order to capture information from a low-level encoder, we add auxiliary supervision branches after certain intermediate layers. The deeply supervised structure makes the training much faster and improves the prediction performance.

The two types of models nicely complement each other. The cropping window model specializes in capturing minute details necessary to discern difficult classes, whereas the whole-image models are able to judge based on the overall context of the image.
