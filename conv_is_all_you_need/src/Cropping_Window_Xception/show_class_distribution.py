from fire import Fire
from lib.config import class_labels
from lib.utils import str_to_labels, debug
import math
import os
import numpy as np
import pandas as pd


def show_status(target_col='Target', **kwargs):
    kwargs = kwargs or {'train_with_hpa': 'data/train_with_hpa.csv'}
    n_samples_list = {}
    for id_, path in kwargs.items():
        anno = pd.read_csv(path, index_col=0)
        n_classes = len(class_labels)
        xs = []
        for target_str in anno[target_col]:
            targets = str_to_labels(target_str)
            x = np.zeros(n_classes, dtype='int')
            x[targets] = 1
            xs.append(x)
        xx = np.array(xs)
        n_samples_list[id_] = np.sum(xx, axis=0)

    cut_summary = pd.DataFrame(
        {
            'organelle': class_labels,
            **n_samples_list,
            # 'pct_samples': n_samples_per_class / len(anno),
            # 'expected_n_samples_in_test': (n_samples_per_class / len(anno) * 11702).round().astype(int),
        },
        index=pd.Index(range(n_classes), name='class_id'),
    )
    save_path = 'tmp/class_distribution.csv'
    cut_summary.to_csv(save_path)
    debug(f"saved to {save_path}")
    print(cut_summary)


if __name__ == '__main__':
    Fire(show_status)
