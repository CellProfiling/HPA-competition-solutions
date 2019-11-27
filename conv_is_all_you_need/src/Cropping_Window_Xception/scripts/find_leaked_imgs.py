import pandas as pd
import sys
from multiprocessing import Pool
import numpy as np
import cv2
from lib.utils import load_image, info, debug, display_imgs
from lib.config import gen_config

config = gen_config()

# use the ladder trick to make it O(n) instead of O(n^2)


def compare_fn(img_record):
    return img_record[0].sum()


def calc_img_hist(img, channel_index=2):
    channels = np.transpose(img, [2, 0, 1])
    return np.array([np.bincount(channel.flatten()) for channel in channels])


def load_one_img(task):
    id_ = task['id_']
    folder = task['folder']
    with_trans = task['with_trans']

    img = load_image(
        id_,
        config,
        resize=(8, 8),
        folder=folder,
        extension='jpg',
        channel=None,
    ) / 255.

    if with_trans:
        img_trans_dict = {
            'original': img,
            # 'rot90': cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            # 'rot180': cv2.rotate(img, cv2.ROTATE_180),
            # 'rot270': cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            # 'transpose': cv2.transpose(img),
            # 'transpose_rot90': cv2.rotate(cv2.transpose(img), cv2.ROTATE_90_CLOCKWISE),
            # 'transpose_rot180': cv2.rotate(cv2.transpose(img), cv2.ROTATE_180),
            # 'transpose_rot270': cv2.rotate(cv2.transpose(img), cv2.ROTATE_90_COUNTERCLOCKWISE),
        }
        return {
            'id_': id_,
            'img': img_trans_dict,
        }
    else:
        return {
            'id_': id_,
            'img': img,
        }


def get_set(path_to_anno, folder, with_trans=False, n_threads=70):
    anno = pd.read_csv(path_to_anno, index_col=0)
    load_list = [{'id_': id_, 'folder': folder, 'with_trans': with_trans} for id_, row in anno.iterrows()]
    with Pool(n_threads) as p:
        result_iter = p.imap(load_one_img, load_list)
        img_list = []
        for i_result, result in enumerate(result_iter):
            info(f"({i_result}/{len(load_list)}) {result['id_']}")
            img_list.append((result['img'], result['id_']))
    # img_list.sort(key=compare_fn)
    return img_list


def comparison_for_one_test(task):
    test_img = task['test_img']
    test_id = task['test_id']
    hpa_trans_set = task['hpa_trans_set']
    identical_pairs = []
    for i_hpa, (hpa_img_trans, hpa_id) in enumerate(hpa_trans_set):
        for trans, hpa_img in hpa_img_trans.items():
            diff_per_channel = np.amin(np.amax(np.abs(test_img - hpa_img), axis=(0, 1)))
            diff_score = np.min(diff_per_channel)
            if diff_score < 0.15:
                identical_pair = {
                    'test_id': test_id,
                    'hpa_id': hpa_id,
                    'diff_score': diff_score,
                    'trans': trans,
                }
                identical_pairs.append(identical_pair)

    return {
        'test_id': test_id,
        'identical_pairs': identical_pairs,
    }


def first_round(
    n_threads=70,
    path_to_test_anno='./data/sample_submission.csv',
    path_to_test_imgs='./data/test_full_size_compressed',
    path_to_hpa_anno='./data/hpa_public_imgs.csv',
    path_to_hpa_imgs='./data/hpa_public_imgs',
):
    test_set = get_set(path_to_test_anno, path_to_test_imgs, n_threads=n_threads)
    hpa_trans_set = get_set(path_to_hpa_anno, path_to_hpa_imgs, with_trans=True, n_threads=n_threads)
    # test_set = get_set('./tmp/test1.csv', './tmp/test1/.', n_threads=n_threads)
    # hpa_trans_set = get_set(
    #     './tmp/test1.csv',
    #     './tmp/test2',
    #     with_trans=True,
    #     n_threads=n_threads,
    # )

    identical_pairss = []
    comparison_list = [
        {
            'test_img': test_img,
            'test_id': test_id,
            'hpa_trans_set': hpa_trans_set
        } for test_img, test_id in test_set
    ]
    with Pool(n_threads) as p:
        result_iter = p.imap(comparison_for_one_test, comparison_list)
        for i_result, result in enumerate(result_iter):
            info(f"Finished ({i_result}/{len(comparison_list)}) {result['test_id']}")
            if len(result['identical_pairs']) > 0:
                debug(f"Found {len(result['identical_pairs'])} identical pairs!")
                identical_pairss.append(result['identical_pairs'])
    identical_pairs = [x for l in identical_pairss for x in l]

    info(f'All done! Found {len(identical_pairs)} pairs.')
    save_path = './tmp/identical_pairs_bk.csv'
    out_df = pd.DataFrame.from_records(identical_pairs).sort_values('diff_score')
    out_df.to_csv(save_path, index=False)
    debug(f'Saved results to {save_path}')


def hires_compare_one(task):
    path_to_test_imgs = './data/test_full_size_compressed'
    path_to_hpa_imgs = './data/hpa_public_imgs'
    test_img = load_image(
        task['test_id'],
        config,
        resize=(16, 16),
        folder=path_to_test_imgs,
        extension='jpg',
        channel=None,
    ) / 255.
    hpa_img = load_image(
        task['hpa_id'],
        config,
        resize=(16, 16),
        folder=path_to_hpa_imgs,
        extension='jpg',
        channel=None,
    ) / 255.
    test_img_no_green = test_img[:, :, [0, 1]]
    hpa_img_no_green = hpa_img[:, :, [0, 1]]
    max_error = np.max(np.abs(test_img_no_green - hpa_img_no_green))
    mean_of_imgs = np.mean(test_img_no_green + hpa_img_no_green) / 2
    out = task.copy()
    out.update({
        'max_error': max_error,
        'mean_of_imgs': mean_of_imgs,
        'relaive_max_error': max_error / mean_of_imgs,
    })
    return out


def hires_compare(
    n_threads=70,
    path_to_test_anno='./data/sample_submission.csv',
    path_to_test_imgs='./data/test_full_size_compressed',
    path_to_hpa_anno='./data/hpa_public_imgs.csv',
    path_to_hpa_imgs='./data/hpa_public_imgs',
):
    pairs_anno = pd.read_csv('./tmp/identical_pairs_bk.csv')
    out_rows = []
    task_list = [row.to_dict() for i_row, row in pairs_anno.iterrows()]
    with Pool(n_threads) as p:
        result_iter = p.imap_unordered(hires_compare_one, task_list)
        for i_result, result in enumerate(result_iter):
            info(f"({i_result}/{len(task_list)}) {result['test_id']}  -  {result['hpa_id']}")
            out_rows.append(result)
    out_anno = pd.DataFrame.from_records(out_rows)
    out_anno = out_anno.sort_values('relaive_max_error')
    out_anno.to_csv('./tmp/identical_pairs.csv', index=False)


def show_top_imgs(save_path='./tmp/identical_pairs.csv'):
    path_to_test_imgs = './data/test_full_size_compressed'
    path_to_hpa_imgs = './data/hpa_public_imgs'
    img_list = []
    out_df = pd.read_csv(save_path)
    df = out_df.head(144)
    for i_row, row in df.iterrows():
        info(f"({i_row}/{len(df)}) {row['test_id']}  -  {row['hpa_id']}")
        test_img = load_image(
            row['test_id'],
            config,
            resize=(512, 512),
            folder=path_to_test_imgs,
            extension='jpg',
            channel=None,
        ) / 255.
        hpa_img = load_image(
            row['hpa_id'],
            config,
            resize=(512, 512),
            folder=path_to_hpa_imgs,
            extension='jpg',
            channel=None,
        ) / 255.
        test_img_resized = cv2.resize(test_img, (16, 16))
        # test_img_resized[:, :, 1] = 0
        hpa_img_resized = cv2.resize(hpa_img, (16, 16))
        # hpa_img_resized[:, :, 1] = 0
        diff_img = test_img_resized - hpa_img_resized
        img_list.append((test_img, f"{row['test_id']} ({test_img.shape}, avg: {test_img.mean():.3f})"))
        img_list.append(
            (
                hpa_img,
                f"{row['hpa_id']}\n({hpa_img.shape}, avg: {hpa_img.mean():.3f}, {row['trans']}, max_error: {row['max_error']:.3f})"
            )
        )
        img_list.append((diff_img / 2 + 0.5, f"test - hpa (max: {diff_img.max():.3f}, min: {diff_img.min():.3f})"))
    display_imgs(img_list, dpi=50, n_cols=27, save_as='./tmp/comparison.jpg')


# identical_pair = {
#     'test_id': test_id,
#     'hpa_id': hpa_id,
#     'max_error': np.max(np.abs(test_img - img)),
#     'mean_error': (test_img - img).mean(),
#     'trans': trans,
# }

if __name__ == '__main__':
    first_round()
    hires_compare()
    show_top_imgs()

# i_test = 0
# i_hpa = 0
# identical_pairs = []
# while i_test < len(test_set) and i_hpa < len(hpa_set):
#     test_img, test_id = test_set[i_test]
#     hpa_img, hpa_id = hpa_set[i_hpa]
#     if np.allclose(test_img, hpa_img, rtol=0, atol=1):
#         identical_pair = {
#             'test_id': test_id,
#             'hpa_id': hpa_id,
#             'error': (test_img - hpa_img).sum(),
#         }
#         debug(f'identical_pair =\n{identical_pair}')
#         identical_pairs.append(identical_pair)
#         i_test +=1
#         i_hpa +=1
#         continue

#     if test_img >

# print(test_set)
# print(hpa_set)
