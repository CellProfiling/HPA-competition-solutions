import cv2
import skimage.io
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import Pool
from config import *
import utils
import matplotlib.pyplot as plt
from collections import defaultdict


def downscale_img_1024(request):
    src_fn, dst_fn = request
    print(src_fn)
    img = cv2.imread(src_fn, flags=cv2.IMREAD_UNCHANGED)
    print(img.shape, img.dtype)
    dst = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
    if len(img.shape) == 3:
        dst = np.max(dst, axis=2)
    cv2.imwrite(dst_fn, dst)


def downscale_img_512(request):
    src_fn, dst_fn = request
    print(src_fn)
    img = cv2.imread(src_fn, flags=cv2.IMREAD_UNCHANGED)
    print(img.shape, img.dtype)
    dst = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    if len(img.shape) == 3:
        dst = np.max(dst, axis=2)
    cv2.imwrite(dst_fn, dst)


def downscale_images_1024(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    requests = []
    pool = Pool(8)

    for fn in list(sorted(os.listdir(src_dir))):
        if not (fn.endswith('tif') or fn.endswith('jpg')):
            print('skip', fn)
            continue
        dst = f'{dst_dir}/{os.path.basename(fn)[:-4]}.png'
        if not os.path.exists(dst):
            requests.append((f'{src_dir}/{fn}', dst))
        # downscale_img_1024(f'{src_dir}/{fn}', f'{dst_dir}/{os.path.basename(fn)}.png')

    for _ in tqdm(pool.imap_unordered(downscale_img_1024, requests), total=len(requests)):
        pass


def downscale_images_512(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    requests = []
    pool = Pool(8)

    for fn in list(sorted(os.listdir(src_dir))):
        if not (fn.endswith('jpg') or fn.endswith('tif')):
            print('skip', fn)
            continue
        dst = f'{dst_dir}/{os.path.basename(fn)[:-4]}.png'
        if not os.path.exists(dst):
            requests.append((f'{src_dir}/{fn}', dst))
        # downscale_img_1024(f'{src_dir}/{fn}', f'{dst_dir}/{os.path.basename(fn)}.png')

    for _ in tqdm(pool.imap_unordered(downscale_img_512, requests), total=len(requests)):
        pass


def check_extra_data_matching_pairs(train_id, extra_id):
    # f, axarr = plt.subplots(4, 4)
    #
    # for c_idx, color in enumerate(COLORS):
    #     fn_train = f'{TEST_DIR}/{train_id}_{color}.png'
    #     fn_extra = f'../input/data_extra/img_512/{extra_id}_{color}.png'
    #     print(fn_train, os.path.exists(fn_train))
    #     print(fn_extra, os.path.exists(fn_extra))
    #
    #     train_img = cv2.imread(fn_train, cv2.IMREAD_GRAYSCALE).astype("uint8")
    #     extra_img = cv2.imread(fn_extra, cv2.IMREAD_UNCHANGED).astype("uint8")
    #
    #     utils.print_stats('train', train_img)
    #     utils.print_stats('extra', extra_img)
    #     utils.print_stats('extra max', np.max(extra_img, axis=2))
    #
    #     axarr[c_idx, 0].imshow(train_img)
    #     axarr[c_idx, 1].imshow(extra_img)
    #     axarr[c_idx, 2].imshow(np.max(extra_img, axis=2))
    #     axarr[c_idx, 3].imshow(np.mean(extra_img, axis=2))
    #
    # plt.show()
    import pandas as pd
    df = pd.read_csv('../input/TestEtraMatchingUnder_259_R14_G12_B10.csv')
    train_mean = defaultdict(list)
    extra_mean = defaultdict(list)

    for r in df.itertuples():
        for c_idx, color in enumerate(COLORS):
            fn_train = f'{TEST_DIR}/{r.Test}_{color}.png'
            # fn_extra = f'../input/data_extra/img_512.old/{r.Extra[len("ENSG00000001497_"):]}_{color}.png'
            fn_extra = f'../input/data_extra/img/{r.Extra[len("ENSG00000001497_"):]}_{color}.jpg'
            print(fn_train, os.path.exists(fn_train))
            print(fn_extra, os.path.exists(fn_extra))

            if not os.path.exists(fn_extra):
                continue

            train_img = cv2.imread(fn_train, cv2.IMREAD_GRAYSCALE).astype("uint8")
            extra_img = cv2.imread(fn_extra, cv2.IMREAD_UNCHANGED).astype("uint8")

            train_mean[color].append(np.mean(train_img))
            extra_mean[color].append(np.mean(np.max(extra_img, axis=2)))
            # extra_mean[color].append(np.mean(extra_img))

    for c_idx, color in enumerate(COLORS):
        # plt.figure()
        plt.scatter(train_mean[color], extra_mean[color], c=color)
    plt.show()


def check_extra_data_sequential_images():
    df = pd.read_csv('../input/data_extra/HPAv18RGBY_WithoutUncertain.csv')
    for name in df['Id']:
        img_id = name.split('_', 1)[1]

        print(name)
        fn = f'../input/data_extra/img_512/{img_id}_green.png'
        img = cv2.imread(fn, cv2.IMREAD_UNCHANGED).astype("uint8")
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    # downscale_images_1024('../input/full_res/train', '../input/full_res/train_1024')
    # downscale_images_1024('../input/full_res/test', '../input/full_res/test_1024')
    downscale_images_512('../input/data_extra/img', '../input/data_extra/img_512')
    # check_extra_data_matching_pairs('7729b27c-bacc-11e8-b2b8-ac1f6b6435d0', '44431_556_B9_1')
    # check_extra_data_sequential_images()
