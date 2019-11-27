# from lib.train import train_all as train
from lib.train import train_folds as train
from time import perf_counter
from datetime import timedelta
from lib.config import gen_config, print_config
from lib.utils import load_config, debug
from fire import Fire
import os


def train_wrapper(config):
    print_config(config)

    t0 = perf_counter()

    debug(f"overwriting soft link ./latest_cwd  -->  {config['_cwd']}")
    os.unlink('./latest_cwd')
    os.symlink(config['_cwd'], './latest_cwd', target_is_directory=True)

    train(config)

    print(f"cwd: {config['_cwd']}")
    print(f"total running time: {str(timedelta(seconds=perf_counter() - t0))}")


def main(do_test=False, job_title='', cwd=None):
    if cwd is None:
        config_overwrites = {
            'job_title': job_title,
            'n_batches_preview': 0,
        }
        config = gen_config(config_overwrites)
    else:
        config = load_config(cwd)

    if do_test:
        test_config = config.copy()
        test_config.update(
            {
                'is_test_run': True,
                'job_title': 'test_run',
                'n_batches_preview': 0,
                'subsampling': 256,
                'submission_subsampling': 32,
                'steps_per_epoch': 1,
                'steps_per_epoch_for_valid': 1,
                'pretraining_n_epochs': 1,
                'n_epochs': 100,
                'lr_scan_n_epochs': 25,
            }
        )
        train_wrapper(test_config)

    if do_test != 'only':
        train_wrapper(config)


if __name__ == '__main__':
    Fire(main)

# train_with_test_run(
#     model_version='basic_iv3_focal_loss',
#     job_title='focal_loss',
#     reduce_lr_on_plateau_patience=4,
#     early_stopping_patience=6,
#     batch_size=32,
# )

# train(
#     pretraining=True,
#     pretraining_n_epochs=2,
#     n_epochs=15,
#     job_title='benchmark__pretraining_old_model',
#     model_version='old',
#     is_test_run=False
# )

# train(
#     pretraining=True,
#     pretraining_n_epochs=3,
#     n_epochs=20,
#     job_title='all_four_channels__new_model__3_15',
#     model_version='new',
# )

# train(lock_inception=False, n_epochs=10, job_title='freeze_inception_average_pooling')
# train(lock_inception=False, n_epochs=10, job_title='benchmark__pretraining_old_model', model_version='old')
