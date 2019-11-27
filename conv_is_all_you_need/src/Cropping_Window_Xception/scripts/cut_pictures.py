from multiprocessing import Pool
from os import makedirs
from os.path import join as pjoin
from traceback import print_exc
import warnings

import cv2
import numpy as np
import pandas as pd
from skimage.io import imsave
from skimage import img_as_ubyte

from lib.config import gen_config, class_labels
from lib.utils import (debug, display_imgs, info, load_img, str_to_labels, warn)


def filter_(n):

    def filter_fixed(row):
        return n in str_to_labels(row['Target'])

    return filter_fixed


def crop_one_id(task):
    try:
        id_ = task['id_']
        group = task['row']['group']
        n_windows = task['row']['n_windows']

        config = task['config']
        n_candidate_windows = config['n_candidate_windows']
        # random_if_no_centroid = config['random_if_no_centroid']
        output_windowed_imgs_path = config['output_windowed_imgs_path']
        output_windowed_imgs_extension = config['output_windowed_imgs_extension']

        rgby_img = load_img(
            id_,
            group=group,
        )

        rgby_thumbnail = cv2.resize(rgby_img, (256, 256))
        green_thumbnail = rgby_thumbnail[:, :, 1].astype(float)

        # the reason we choose centers first is to ensure that every possible "interest point"
        # in the picture will get equal chance of being captured. if we choose from all contained windows
        # with equal chance, then the interest points around the edge of the picture will get
        # less chance of being captured.
        random_centers = np.random.rand(n_candidate_windows, 2) * (1 - config['win_size'] / 2) + config['win_size'] / 4
        random_centers = np.minimum(random_centers, 1 - config['win_size'] / 2)
        random_centers = np.maximum(random_centers, config['win_size'] / 2)
        windows = [
            np.array(
                [
                    x - config['win_size'] / 2,
                    y - config['win_size'] / 2,
                    x + config['win_size'] / 2,
                    y + config['win_size'] / 2,
                ]
            ) for x, y in random_centers
        ]

        img_size = green_thumbnail.shape[0]
        greenest_windows = []
        for _ in range(n_windows):
            green_totals = []
            for window in windows:
                left, top, right, bottom = (window * img_size).astype(int)
                green_totals.append(np.sum(green_thumbnail[top:bottom, left:right]))
            greenest_window = windows.pop(np.argmax(green_totals))
            left, top, right, bottom = (greenest_window * img_size).astype(int)
            green_thumbnail[top:bottom, left:right] *= 0.6
            greenest_windows.append(greenest_window)
            if len(windows) == 0:
                break

        img_size = rgby_img.shape[0]
        # dot_size = max(round(img_size / 64), 1)
        img_ids = []
        source_img_ids = []
        for i_window, window in enumerate(greenest_windows):
            left, top, right, bottom = (window * img_size).astype(int)
            if output_windowed_imgs_path is not None:
                cropped_img = rgby_img[top:bottom, left:right]
                image_id = f'{id_}_{i_window}'
                img_ids.append(image_id)
                source_img_ids.append(id_)
                for i_channel, channel in enumerate(['red', 'green', 'blue', 'yellow']):
                    img_filename = f'{image_id}_{channel}{output_windowed_imgs_extension}'
                    cropped_img_channel = cropped_img[:, :, i_channel]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cropped_img_channel = img_as_ubyte(cropped_img_channel)
                    imsave(pjoin(output_windowed_imgs_path, img_filename), cropped_img_channel)

        out_df = pd.DataFrame(np.array(greenest_windows), columns=['left', 'top', 'right', 'bottom'])
        out_df.index = pd.Index(img_ids, name='img_id')
        out_df['source_img_id'] = source_img_ids

        return {
            'id_': id_,
            'df': out_df,
            # 'blue_channel': (
            #     blue_channel,
            #     '\n'.join([
            #         f'{id_}',
            #         f'pct = {pct}',
            #         f'avg_of_brightest_5% = {avg_of_brightest}',
            #     ]),
            # ),
            # 'thresholded_img': (
            #     thresholded_img,
            #     '\n'.join([
            #         f'{id_}',
            #         f'ret = {ret}',
            #         f'th = {th}',
            #     ]),
            # ),
            # 'labeled_img': (
            #     labeled_img,
            #     '\n'.join([
            #         f'{id_}',
            #         f'ret = {ret}',
            #         f'greenest_windows = {greenest_windows}',
            #     ]),
            # ),
        }
    except Exception as e:
        debug(f'Error in processing {id_}')
        print_exc()
        return {
            'id_': id_,
            'df': pd.DataFrame({
                'img_ids': id_,
                'left': ['ERROR'],
                'top': [str(e)],
            }).set_index('img_ids'),
        }


def crop(config):
    if config['output_windowed_imgs_path'] is not None:
        makedirs(config['output_windowed_imgs_path'], exist_ok=True)

    if type(config['set_n_windows']) is str:
        set_n_windows_anno = pd.read_csv(config['set_n_windows'], index_col=0)

        n_classes = 28

        xs = []
        for target_str in set_n_windows_anno['Target']:
            targets = str_to_labels(target_str)
            x = np.zeros(n_classes, dtype='int')
            x[targets] = 1
            xs.append(x)
        xx = np.array(xs)
        n_samples_per_class = np.sum(xx, axis=0)
        cut_summary = pd.DataFrame(
            {
                'organelle': class_labels,
                'n_samples': n_samples_per_class,
                'n_windows': np.round(1500 / n_samples_per_class).astype(int) + 1
            },
            index=range(n_classes),
        )
        print(cut_summary)
        estimated_n_windows = np.sum(cut_summary['n_samples'].values * cut_summary['n_windows'].values)
        print(f'estimated_n_windows = {estimated_n_windows}')

    def determine_n_windows_fn(id_):
        if type(config['set_n_windows']) is str:
            targets = str_to_labels(set_n_windows_anno.loc[id_, 'Target'])
            n_windows = np.max(cut_summary.iloc[targets]['n_windows'].values)
            return n_windows
        else:
            return config['set_n_windows']

    anno = config['anno'].copy()
    anno['n_windows'] = [determine_n_windows_fn(id_) for id_ in anno.index]

    crop_task_list = [{
        'id_': id_,
        'row': row,
        'config': config,
    } for id_, row in anno.iterrows()]

    with Pool(config['n_threads']) as p:
        result_iter = p.imap_unordered(crop_one_id, crop_task_list)

        result_list = []
        for i_result, result in enumerate(result_iter):
            info(f"({i_result}/{len(crop_task_list)}) {result['id_']}  ->  ({len(result['df'])})")
            result_list.append(result)

    if config['output_windowed_imgs_path'] is not None:
        windowed_anno = pd.concat([x['df'] for x in result_list])
        print(windowed_anno)
        if 'ERROR' in windowed_anno['left']:
            warn(f'There were errors!')
        windowed_anno.to_csv(config['output_windowed_anno_csv_path'])

    def save_collage(field):
        display_imgs(
            [x[field] for x in result_list],
            save_as=pjoin(config['collage_output_path'], f"{config['run_tag']}-0-{field}.jpg")
        )

    if config['collage_output_path'] is not None:
        save_collage('blue_channel')
        save_collage('thresholded_img')
        save_collage('labeled_img')
        save_collage('minimap')


def main():
    # train_config = gen_config()

    # -- train -----------------------------------------------------------------------

    # validation has to be cut the same way as the test is

    anno = pd.read_csv('./data/train_with_hpa.csv', index_col=0)

    crop(
        {
            'anno': anno,
            'collage_output_path': None,
            'n_threads': 80,
            'random_if_no_centroid': True,
            'run_tag': 'train',
            'set_n_windows': 10,
            'n_candidate_windows': 60,
            'output_windowed_anno_csv_path': 'data/train_windowed_0.4.csv',
            'output_windowed_imgs_path': 'data/train_windowed_0.4',
            'output_windowed_imgs_extension': '.jpg',
            'win_size': 0.4,
        }
    )

    # -- test ------------------------------------------------------------------------

    anno = pd.read_csv('./data/sample_submission.csv', index_col=0)
    anno['group'] = 'test_full_size'

    crop(
        {
            'anno': anno,
            'collage_output_path': None,
            'n_threads': 80,
            'random_if_no_centroid': True,
            'run_tag': 'train',
            'set_n_windows': 10,
            'n_candidate_windows': 60,
            'output_windowed_anno_csv_path': './data/test_windowed_0.4.csv',
            'output_windowed_imgs_path': './data/test_windowed_0.4',
            'output_windowed_imgs_extension': '.jpg',
            'win_size': 0.4,
        }
    )

    # -- train -----------------------------------------------------------------------

    # validation has to be cut the same way as the test is

    anno = pd.read_csv('./data/train_with_hpa.csv', index_col=0)

    crop(
        {
            'anno': anno,
            'collage_output_path': None,
            'n_threads': 80,
            'random_if_no_centroid': True,
            'run_tag': 'train',
            'set_n_windows': 10,
            'n_candidate_windows': 60,
            'output_windowed_anno_csv_path': 'data/train_windowed_0.3.csv',
            'output_windowed_imgs_path': 'data/train_windowed_0.3',
            'output_windowed_imgs_extension': '.jpg',
            'win_size': 0.3,
        }
    )

    # -- test ------------------------------------------------------------------------

    anno = pd.read_csv('./data/sample_submission.csv', index_col=0)
    anno['group'] = 'test_full_size'

    crop(
        {
            'anno': anno,
            'collage_output_path': None,
            'n_threads': 80,
            'random_if_no_centroid': True,
            'run_tag': 'train',
            'set_n_windows': 10,
            'n_candidate_windows': 60,
            'output_windowed_anno_csv_path': './data/test_windowed_0.3.csv',
            'output_windowed_imgs_path': './data/test_windowed_0.3',
            'output_windowed_imgs_extension': '.jpg',
            'win_size': 0.3,
        }
    )
    # anno = pd.read_csv('./data/sample_submission.csv', index_col=0)

    # crop(
    #     anno.index.values,
    #     10,
    #     anno,
    #     config,
    #     random_if_no_centroid=True,
    #     small_img_path='./data/test',
    #     large_img_path='./data/test_full_size',
    #     windows_imgs_path='./data/test_windowed',
    #     anno_csv_path='./data/test_windowed.csv',
    #     subsampled_anno_csv_path=None,
    # )

    # -- main ------------------------------------------------------------------------

    #     # [
    #     #     '4b179d38-bb9a-11e8-b2b9-ac1f6b6435d0',
    #     #     '056fe7a6-bb9d-11e8-b2b9-ac1f6b6435d0',
    #     #     '0662c046-bb9f-11e8-b2b9-ac1f6b6435d0',
    #     #     '1595a842-bb9c-11e8-b2b9-ac1f6b6435d0',
    #     #     '1d2019c8-bbae-11e8-b2ba-ac1f6b6435d0',
    #     #     '2ba53350-bba6-11e8-b2ba-ac1f6b6435d0',
    #     #     '4ec8b86a-bba2-11e8-b2b9-ac1f6b6435d0',
    #     #     '886e6e5e-bbaf-11e8-b2ba-ac1f6b6435d0',
    #     #     '92b1c690-bbc3-11e8-b2bc-ac1f6b6435d0',
    #     #     '987794f0-bbba-11e8-b2ba-ac1f6b6435d0',
    #     #     '99934838-bba7-11e8-b2ba-ac1f6b6435d0',
    #     #     'a1c2b8d0-bbad-11e8-b2ba-ac1f6b6435d0',
    #     #     'b5f3ebfc-bba4-11e8-b2ba-ac1f6b6435d0',
    #     #     'befd3196-bbb2-11e8-b2ba-ac1f6b6435d0',
    #     #     'c77f25e2-bbab-11e8-b2ba-ac1f6b6435d0',
    #     #     'cc706872-bba6-11e8-b2ba-ac1f6b6435d0',
    #     #     'd9ab4ffe-bbae-11e8-b2ba-ac1f6b6435d0',
    #     #     'da60ebd4-bb9a-11e8-b2b9-ac1f6b6435d0',
    #     #     'e39dc4dc-bba5-11e8-b2ba-ac1f6b6435d0',
    #     # ],
    #     [
    #         '0798a8c8-bad6-11e8-b2b9-ac1f6b6435d0',
    #         '11fffb44-bad2-11e8-b2b8-ac1f6b6435d0',
    #         '2ea1345a-bacf-11e8-b2b8-ac1f6b6435d0',
    #         '40ef5af2-bace-11e8-b2b8-ac1f6b6435d0',
    #     ],
    #     10,
    #     # './data/train.csv',
    #     config,
    #     run_tag=f'errors',
    #     show_collage=True,
    #     small_img_path='./data/test',
    #     large_img_path='./data/test_full_size',
    #     windows_imgs_path='./data/test_windowed',
    #     anno_csv_path='./data/test_windowed.csv',
    # )

    # random_state = random.randint(0, 65535)
    # crop(anno.sample(28, random_state=random_state), run_tag=str(random_state))
    # selected_class = 27
    # crop(
    #     anno.loc[anno.apply(filter_(selected_class), axis=1)].index.values,
    #     './data/train.csv',
    #     anno,
    #     config,
    #     random_if_no_centroid=False,
    #     run_tag=f'{selected_class}_only',
    #     show_collage=True,
    # )


if __name__ == '__main__':
    main()
