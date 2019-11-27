from lib.utils import preview_generator, gen_even_batches, batching_row_gen, randomize_and_loop, str_to_labels, load_img
from lib.config import gen_config
import numpy as np
from fire import Fire
import pandas as pd


def row_has_class(class_id):

    def row_has_class_i(row):
        target = str_to_labels(row['Target'])
        return int(class_id) in target

    return row_has_class_i


def img_gen_from_anno_gen(
    anno_gen,
    config,
    group=None,
    target_col=None,
):
    while True:
        anno_batch_df = next(anno_gen)
        batch_x_list = []
        batch_y_list = []
        for id_, row in anno_batch_df.iterrows():
            img = load_img(
                id_,
                resize=None,
                channels=config['channels'],
                group=group or row['group'],
            )

            labels = str_to_labels(row[target_col])
            y_vec = np.array([1 if class_id in labels else 0 for class_id in config['class_ids']])

            batch_y_list.append(y_vec)
            batch_x_list.append(img / 255.)

        batch_x = np.array(batch_x_list, dtype=np.float32)
        batch_y = np.array(batch_y_list, dtype=np.float32)

        yield batch_x, batch_y


def main(class_id):
    config = gen_config()
    anno = pd.read_csv('data/train_with_hpa.csv', index_col=0)
    anno_only_class_i = anno.loc[anno.apply(row_has_class(class_id), axis=1)]

    preview_generator(
        img_gen_from_anno_gen(
            batching_row_gen(randomize_and_loop(anno_only_class_i), 64),
            config=config,
            target_col='Target',
        ),
        config,
        filename_prefix=f"class_{class_id}",
        n_batches=3,
    )


if __name__ == '__main__':
    Fire(main)
