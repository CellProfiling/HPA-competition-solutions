from os.path import join as pjoin

import datetime
import random
from colors import color

class_labels = [
    'Nucleoplasm',  # 0
    'Nuclear membrane',  # 1
    'Nucleoli',  # 2
    'Nucleoli fibrillar center',  # 3
    'Nuclear speckles',  # 4
    'Nuclear bodies',  # 5
    'Endoplasmic reticulum',  # 6
    'Golgi apparatus',  # 7
    'Peroxisomes',  # 8
    'Endosomes',  # 9
    'Lysosomes',  # 10
    'Intermediate filaments',  # 11
    'Actin filaments',  # 12
    'Focal adhesion sites',  # 13
    'Microtubules',  # 14
    'Microtubule ends',  # 15
    'Cytokinetic bridge',  # 16
    'Mitotic spindle',  # 17
    'Microtubule organizing center',  # 18
    'Centrosome',  # 19
    'Lipid droplets',  # 20
    'Plasma membrane',  # 21
    'Cell junctions',  # 22
    'Mitochondria',  # 23
    'Aggresome',  # 24
    'Cytosol',  # 25
    'Cytoplasmic bodies',  # 26
    'Rods & rings',  # 27
]

default_config = {
    'i_fold': None,
    'augmentation_at_training': True,
    'batch_size': 64,
    'class_ids': list(range(28)),
    'class_labels': class_labels,
    'cuda_visible_devices': None,
    '_cwd': None,
    'do_lr_scanning': False,
    'early_stopping_patience': 6,
    'epsilon': 1e-7,
    'frac_of_validation_samples': 0.15,
    'n_folds': 5,
    'img_extension': 'jpg',
    '_input_shape': None,
    'is_test_run': False,
    '_job_id': None,
    'job_title': None,
    'lr_scan_n_epochs': 100,
    'max_queue_size': 0,
    'metric': 'val_macro_f1_metric',
    'n_batches_preview': 0,
    'n_channels': 4,
    'channels': ['red', 'green', 'blue', 'yellow'],
    '_n_classes': None,
    'n_epochs': 500,
    'path_to_test_anno': 'data/test_windowed_0.4.csv',
    'path_to_test': 'data/test_windowed_0.4',
    'path_to_train_anno': 'data/train_with_hpa.csv',
    'path_to_train': 'data/train_windowed_0.4',
    'path_to_train_windowed_anno': 'data/train_windowed_corrected_0.4.csv',
    'path_to_train_anno_cache': None,
    'path_to_train_windowed_anno_cache': None,
    'path_to_valid_anno_cache': None,
    'path_to_valid_windowed_anno_cache': None,
    'path_to_valid': 'data/train_windowed_0.4',
    'path_to_valid_windowed_anno': 'data/train_windowed_corrected_0.4.csv',
    'predict_batch_size': 64,
    'pretraining_n_epochs': 1,
    '_random_state': None,
    'reduce_lr_on_plateau_patience': 3,
    'score_threshold': 0.5,
    'size': 384,  # TODO: can we increse the size? or use a better resizing algorithm than interpolation?
    'source_img_size': 960,
    'starting_lr': 1e-3,
    'steps_per_epoch': 500,
    'steps_per_epoch_for_valid': 10,
    'submission_subsampling': None,
    'subsampling': None,
    'aug_negative_control': 0.1,
    'aug_window_jittering': 4,
    'verbose': 1,
}


def gen_config(config_overwrites={}, save_as_json=True):
    config = default_config.copy()
    config.update(config_overwrites)

    config['_n_classes'] = len(config['class_ids'])
    config['_input_shape'] = [config['size'], config['size'], config['n_channels']]

    if config['_random_state'] is None:
        config['_random_state'] = random.randint(0, 65536)

    if config['_job_id'] is None:
        config['_job_id'] = (
            '_'.join(
                [
                    f"{datetime.datetime.now().strftime('%y%m%d-%H%M%S_')}",
                    f"P{config['pretraining_n_epochs']}T{config['n_epochs']}",
                    f"{config['job_title']}" if config['job_title'] is not None else "",
                ]
            )
        )

    if config['_cwd'] is None:
        config['_cwd'] = pjoin('./working', config['_job_id'])

    if config['path_to_train_anno_cache'] is None:
        config['path_to_train_anno_cache'] = pjoin(config['_cwd'], 'train_anno.csv')

    if config['path_to_valid_anno_cache'] is None:
        config['path_to_valid_anno_cache'] = pjoin(config['_cwd'], 'valid_anno.csv')

    if config['path_to_train_windowed_anno_cache'] is None:
        config['path_to_train_windowed_anno_cache'] = pjoin(config['_cwd'], 'train_windowed_anno.csv')

    if config['path_to_valid_windowed_anno_cache'] is None:
        config['path_to_valid_windowed_anno_cache'] = pjoin(config['_cwd'], 'valid_windowed_anno.csv')

    return config


def print_config(config):
    print('*' * 80)
    for k in config.keys():
        if k not in default_config.keys():
            print(color(f'{k}: ', fg='cyan', bg='black'), end='')
        else:
            print(f'{k}: ', end='')

        if default_config.get(k) == config[k]:
            print(config[k])
        else:
            print(color(config[k], fg='yellow', bg='black', style='bold'))
    print('*' * 80)
