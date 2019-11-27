import numpy as np
from ..utils import gen_even_batches, load_image, augment, str_to_labels


class DataLoader:

    def __init__(
        self,
        anno_gen,
        batch_size,
        n_batches_per_epoch,
        config,
        folder=None,
        extension=None,
        do_augmentation=True,
        target_col='Target',
    ):
        self.anno_gen = anno_gen
        self.batch_size = batch_size
        self.n_batches_per_epoch = n_batches_per_epoch
        self.config = config
        self.folder = folder
        self.extension = extension
        self.do_augmentation = do_augmentation
        self.target_col = target_col

    def __iter__(self):
        for i_batch in range(self.n_batches_per_epoch):
            anno_batch_df = next(self.anno_gen)
            batch_x_list = []
            batch_y_list = []
            for id_, row in anno_batch_df.iterrows():
                image = load_image(
                    id_,
                    resize=self.config['_input_shape'],
                    channel=None,
                    folder=self.folder or row.get('folder'),
                    extension=self.extension or row.get('extension'),
                )
                # TODO: augmentation should be on the original resolution (512 x 512) instead of the 299 x 299
                # TODO: need a way to better pick out windows that actually *have* centrosomes/MOC
                if self.do_augment:
                    image = augment(image)
                labels = str_to_labels(row[self.target_col])
                y_vec = np.array([1 if class_id in labels else 0 for class_id in self.config['class_ids']])
                batch_y_list.append(y_vec)
                batch_x_list.append(image / 255.)

            batch_x = np.array(batch_x_list)
            batch_y = np.array(batch_y_list)

            yield np.array(batch_x, np.float32), batch_y
