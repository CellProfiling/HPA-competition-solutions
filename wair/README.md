# WAIR

## Model description

The solution is mainly based on CNN models. Seven models derived from densenet121 (3 variants) [1], densenet169 [1], ibn_densenet121 [2], se-resnextnet50 [3]and xception [4] were trained and evaluated on image data provided by Kaggle and Human Protein Atlas (HPA) websites.

To train each model, Adamax [5] was used as optimizer and a cycle learning rate [6] with warm up was set to schedule. Independent models were finally ensembled through averaging the predicated probability for each class from each model.

The public tools involved are pytorch, opencv and scikit-learn. It takes 20-30 hours to train each model, and the full training process will take a week, using four Nvidia GTX 1080Ti GPUs.

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
