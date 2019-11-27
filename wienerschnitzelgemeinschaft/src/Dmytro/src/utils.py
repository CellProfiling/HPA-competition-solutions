import math
import pickle
import time
import random
from contextlib import contextmanager

import skimage.io
import skimage.transform
from skimage.transform import SimilarityTransform, AffineTransform
import numpy as np
import matplotlib.pyplot as plt

import torch.optim.lr_scheduler

class CosineAnnealingLRWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr,
    :math:`T_{mult}` is the multiplicative factor of T_max and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}\cdot T_{mult}}\pi))
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        T_mult (float): Multiplicative factor of T_max. Default: 2.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=2):
        self.T_max = T_max
        self.Ti = T_max
        self.eta_min = eta_min
        self.T_mult = T_mult
        self.cycle = 0
        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None, metrics=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            if epoch == self.Ti:
                epoch = 0
                self.cycle += 1
        else:
            self.cycle = int(math.floor(math.log(epoch / self.T_max * (self.T_mult - 1) + 1, self.T_mult)))
            epoch -= sum([self.T_max * self.T_mult ** x for x in range(self.cycle)])
        self.last_epoch = epoch
        self.Ti = self.T_max * self.T_mult ** self.cycle
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.Ti)) / 2
                for base_lr in self.base_lrs]


def check_CosineAnnealingLRWithRestarts():
    conv1 = torch.nn.Conv2d(1, 1, 1)
    opt = torch.optim.SGD([{'params': conv1.parameters()}], lr=0.05)
    scheduler = CosineAnnealingLRWithRestarts(opt, T_max=8, T_mult=1.2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8)
    epochs = list(range(100))
    lr = []
    for e in epochs:
        scheduler.step(e)
        lr.append(scheduler.get_lr())

    plt.plot(epochs, lr)
    plt.show()


def crop_edge(img, x, y, w, h, mode='edge'):
    img_w = img.shape[1]
    img_h = img.shape[0]

    if x >= 0 and y >= 0 and x + w <= img_w and y + h < img_h:
        return img[int(y):int(y + h), int(x):int(x + w)].astype('float32') / 255.0

    tform = SimilarityTransform(translation=(x, y))
    return skimage.transform.warp(img, tform, mode=mode, output_shape=(h, w))


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l) // n * n + n - 1, n):
        if len(l[i:i + n]):
            yield l[i:i + n]


def lock_layers_until(model, first_trainable_layer, verbose=False):
    found_first_layer = False
    for layer in model.layers:
        if layer.name == first_trainable_layer:
            found_first_layer = True

        if verbose and found_first_layer and not layer.trainable:
            print('Make layer trainable:', layer.name)
            layer.trainable = True

        layer.trainable = found_first_layer


def get_image_crop(full_rgb, rect, scale_rect_x=1.0, scale_rect_y=1.0,
                   shift_x_ratio=0.0, shift_y_ratio=0.0,
                   angle=0.0, out_size=299):
    center_x = rect.x + rect.w / 2
    center_y = rect.y + rect.h / 2
    size = int(max(rect.w, rect.h))
    size_x = size * scale_rect_x
    size_y = size * scale_rect_y

    center_x += size * shift_x_ratio
    center_y += size * shift_y_ratio

    scale_x = out_size / size_x
    scale_y = out_size / size_y

    out_center = out_size / 2

    tform = AffineTransform(translation=(center_x, center_y))
    tform = AffineTransform(rotation=angle * math.pi / 180) + tform
    tform = AffineTransform(scale=(1 / scale_x, 1 / scale_y)) + tform
    tform = AffineTransform(translation=(-out_center, -out_center)) + tform
    return skimage.transform.warp(full_rgb, tform, mode='edge', order=1, output_shape=(out_size, out_size))


def crop_zero_pad(img, x, y, w, h):
    img_w = img.shape[1]
    img_h = img.shape[0]

    if x >= 0 and y >= 0 and x + w <= img_w and y + h < img_h:
        return img[int(y):int(y + h), int(x):int(x + w)]
    else:
        res = np.zeros((h, w)+img.shape[2:], dtype=img.dtype)
        x_min = int(max(x, 0))
        y_min = int(max(y, 0))
        x_max = int(min(x + w, img_w))
        y_max = int(min(y + h, img_h))
        res[y_min - y:y_max-y, x_min - x:x_max-x] = img[y_min:y_max, x_min:x_max]
        return res


def overlapped_crops_shape(img, crop_w, crop_h, overlap):
    img_h, img_w = img.shape[:2]
    n_h = int(np.ceil((img_h + overlap/2 - 1) / (crop_h - overlap)))
    n_w = int(np.ceil((img_w + overlap/2 - 1) / (crop_w - overlap)))
    return [n_h, n_w]


def generate_overlapped_crops_with_positions(img, crop_w, crop_h, overlap):
    n_h, n_w = overlapped_crops_shape(img, crop_w, crop_h, overlap)

    res = np.zeros((n_w*n_h, crop_h, crop_w, ) + img.shape[2:], dtype=img.dtype)
    positions = []

    for i_h in range(n_h):
        for i_w in range(n_w):
            x = -overlap // 2 + i_w * (crop_w - overlap)
            y = -overlap // 2 + i_h * (crop_h - overlap)
            res[i_h * n_w + i_w] = crop_zero_pad(img, x, y, crop_w, crop_h)
            positions.append((x, y, crop_w, crop_h))

    return res, positions, n_h, n_w


def generate_overlapped_crops(img, crop_w, crop_h, overlap):
    return generate_overlapped_crops_with_positions(img, crop_w, crop_h, overlap)[0]


def rand_or_05():
    if random.random() > 0.5:
        return random.random()
    return 0.5


def rand_scale_log_normal(mean_scale, one_sigma_at_scale):
    """
    Generate a distribution of value at log  scale around mean_scale

    :param mean_scale:
    :param one_sigma_at_scale: 67% of values between  mean_scale/one_sigma_at_scale .. mean_scale*one_sigma_at_scale
    :return:
    """

    log_sigma = math.log(one_sigma_at_scale)
    return mean_scale*math.exp(random.normalvariate(0.0, log_sigma))


def print_stats(title, array):
    if len(array):
        print('{} shape:{} dtype:{} min:{} max:{} mean:{} median:{}'.format(
            title,
            array.shape,
            array.dtype,
            np.min(array),
            np.max(array),
            np.mean(array),
            np.median(array)
        ))
    else:
        print(title, 'empty')


def nonzero_crop(mask):
    """
    Crop mask to keep only non zero areas

    :param mask: mask to crop
    :return: crop, (row_offset, col_offset)
    """
    rows_non_zero = mask.sum(axis=1).nonzero()[0]
    cols_non_zero = mask.sum(axis=0).nonzero()[0]
    crop = mask[rows_non_zero[0]:rows_non_zero[-1] + 1, cols_non_zero[0]:cols_non_zero[-1] + 1]
    return crop.copy(), (rows_non_zero[0], cols_non_zero[0])


def transform_crop(crop: np.ndarray, crop_offset, transform: AffineTransform, output_shape):
    # src_x_min = crop_offset[1]
    # src_x_max = crop_offset[1] + crop.shape[1]
    # src_y_min = crop_offset[0]
    # src_y_max = crop_offset[0] + crop.shape[0]
    # src_edges = np.array([
    #     [src_x_min, src_y_min],
    #     [src_x_min, src_y_max],
    #     [src_x_max, src_y_min],
    #     [src_x_max, src_y_max]])
    # dst_edges = transform(src_edges)

    tform = transform + skimage.transform.AffineTransform(translation=(-crop_offset[1], -crop_offset[0]))
    return skimage.transform.warp(crop, tform,
                                  mode='constant', order=0, output_shape=output_shape)


def test_transform_crop():
    mask = np.zeros((128, 128))
    mask[20:30, 40:45] = 1
    mask[25:30, 45:50] = 1
    mask[31, 50] = 1

    plt.imshow(mask)
    # plt.show()

    transform = skimage.transform.AffineTransform(translation=(-20, -30)) + \
                skimage.transform.AffineTransform(scale=(0.7, 1.2), rotation=1.0, shear=0.1)

    output_shape = (256, 256)
    warp_full = skimage.transform.warp(mask, transform, mode='constant', order=0, output_shape=output_shape)

    plt.figure()
    plt.imshow(warp_full)

    crop, crop_offset = nonzero_crop(mask)
    plt.figure()
    plt.imshow(crop)

    plt.figure()
    warp_from_crop = transform_crop(crop, crop_offset, transform, output_shape)
    plt.imshow(warp_from_crop)

    assert np.max(np.abs(warp_full - warp_from_crop)) == 0
    # plt.show()


def combine_tiled_predictions(model, img, preprocess_input, crop_size, channels, overlap):
    # generate overlapped set of crops
    X, src_positions = generate_overlapped_crops_with_positions(img, crop_w=crop_size, crop_h=crop_size, overlap=overlap)
    tiles_rows, tiles_cols = overlapped_crops_shape(img, crop_w=crop_size, crop_h=crop_size, overlap=overlap)

    y = model.predict(preprocess_input(X), batch_size=1, verbose=1)

    predict_size_cropped = crop_size - overlap

    # + overlap_predict_pixels//2 due to border crops covering overlap of black bands not overlap/2
    res = np.zeros((tiles_rows*predict_size_cropped,
                    tiles_cols*predict_size_cropped, channels))

    for i in range(y.shape[0]):
        y_cur = y[i]
        predicted_cropped = y_cur[overlap//2:-overlap//2, overlap//2:-overlap//2]

        tile_row = i // tiles_cols
        tile_col = i % tiles_cols
        row = tile_row * predict_size_cropped
        col = tile_col * predict_size_cropped

        res[row:row + predict_size_cropped, col:col + predict_size_cropped, :] = predicted_cropped

    # strip extra black borders from the last tiles
    expected_predict_rows = img.shape[0]
    expected_predict_cols = img.shape[1]

    return res[:expected_predict_rows, :expected_predict_cols, :]


def test_combine_tiled_predictions():
    class TestModel:
        def predict(self, X: np.ndarray, batch_size, verbose):
            res = X.copy()

            # damage around border
            border = 24
            res[:, border] /= 2
            res[:, -border:] /= 2

            res[:, :, :border] /= 2
            res[:, :, -border:] /= 2

            return res

    model = TestModel()

    for img_size in [(64, 65), (255, 255), (256, 256), (257, 257),  (512, 512), (60, 600), (1024, 1024), (2000, 2000)]:
        channels = 3
        img_shape = img_size+(channels,)

        img = np.ones(img_shape)  # np.random.rand(*img_shape)
        res = combine_tiled_predictions(model,
                                        img,
                                        preprocess_input=lambda x: x,
                                        crop_size=256,
                                        channels=channels,
                                        overlap=64)
        expected = img
        error = np.max(np.abs(res - expected))
        assert error < 1e-6

        img = np.random.rand(*img_shape)
        res = combine_tiled_predictions(model,
                                        img,
                                        preprocess_input=lambda x: x*2,
                                        crop_size=256,
                                        channels=channels,
                                        overlap=64)
        expected = img*2
        error = np.max(np.abs(res - expected))
        assert error < 1e-6


if __name__ == '__main__':
    pass

    check_CosineAnnealingLRWithRestarts()
    #
    # test_transform_crop()
    # test_combine_tiled_predictions()
    # test_chunks()
    #
    # img = skimage.io.imread('../train/ALB/img_00003.jpg')
    # print(img.shape)
    #
    # with timeit_context('Generate crops'):
    #     crop_edge(img, 10, 10, 400, 400)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure(1)
    # plt.subplot(221)
    # plt.imshow(img)
    # plt.subplot(222)
    # plt.imshow(crop_edge(img, 1280-200, 720-200, 400, 400, mode='edge'))
    # plt.subplot(223)
    # plt.imshow(crop_edge(img, 1280 - 200, 720 - 200, 400, 400, mode='wrap'))
    # plt.subplot(224)
    # plt.imshow(crop_edge(img, 1280 - 200, 720 - 200, 400, 400, mode='reflect'))
    #
    # plt.show()
