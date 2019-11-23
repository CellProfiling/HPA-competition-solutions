# Model description [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A CNN multi-label classification model to predict labels of each sample.

Multi-Label Stratification was used to split the dataset into a training set and a validation set. The performance of the model was estimated using focal loss over the validation set. A Densenet121 acts as the backbone of the model. The GlobalMaxPool and GlobalAvgPool layers of the final CNN feature map were concatenated before being fed to two fully connected layers to calculate the probability of each class.

Image augmentation was performed by flipping the image, rotating it 90 degrees, and randomly cropping out 1024 ⨉ 1024 pixel (px) patches. To improve the model’s predictive power, multiple random crops were taken, and the maximum probability among them was calculated when predicting on the test set.

A combined loss function of focal loss, Lovasz loss, and hard example log loss was used for training the model. It was optimized using an Adam optimizer with a step learning rate of [30, 15, 7.5, 3, 1] ⨉ 1e-5 for [25, 5, 5, 5, 5] epochs respectively. The output was thresholded using the ratio of labels in the training set.

Using this method, the CNN reached 0.565 Macro F1 on the test set when averaging predictions from 5 folds.
