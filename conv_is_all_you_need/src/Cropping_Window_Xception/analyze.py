from lib.steps import after_training_for_folds
from lib.config import print_config
from lib.utils import load_config, debug
import os
import fire


def main(cwd):
    config = load_config(cwd)

    # os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_visible_devices'] or '0,1,2,3,4,5,6,7'

    # avail_gpus = GPUtil.getAvailable(limit=100)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, avail_gpus))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    debug(f"os.environ['CUDA_VISIBLE_DEVICES'] = {os.environ['CUDA_VISIBLE_DEVICES']}")

    print_config(config)

    after_training_for_folds(config)


if __name__ == '__main__':
    fire.Fire(main)
