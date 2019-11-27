import math
import os
from os.path import join as pjoin
import random
from time import sleep

import GPUtil
import pandas as pd
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import (BatchNormalization, Dense, GlobalAveragePooling2D, Input)
from keras.models import Model, clone_model, model_from_yaml
from keras.optimizers import Adam

from .callbacks import MetricOnAll, ModelCheckpoint, TimeLogger
from .steps import after_training
from .config import print_config
from .utils import (
    chart, data_gen_from_anno_gen, debug, format_macro_f1_details, gen_cwd_slash, gen_even_batches, info, np_macro_f1,
    preview_generator
)
from .utils_heavy import gen_macro_f1_metric


# lr_scanning
def lr_scanning(model, train_generator, config, begin_lr=1e-6, end_lr=10):
    info('-- Scan LR --')

    def cwd_slash(fn):
        return pjoin(config['_cwd'], fn)

    factor_per_step = math.exp((math.log(end_lr) - math.log(begin_lr)) / config['lr_scan_n_epochs'])
    tmp_model = clone_model(model)
    tmp_model.set_weights(model.get_weights())
    tmp_model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=begin_lr),
        metrics=['accuracy', gen_macro_f1_metric(config)],
    )
    tmp_model.summary()
    tmp_model.fit_generator(
        train_generator,
        steps_per_epoch=1,
        # epochs=n_batches_in_train,
        epochs=config['lr_scan_n_epochs'],
        verbose=config['verbose'],
        max_queue_size=config['max_queue_size'],
        callbacks=[
            LearningRateScheduler(lambda _, lr: lr * factor_per_step, verbose=config['verbose']),
            CSVLogger(cwd_slash('lr_scan.csv'), append=True),
        ],
    )

    lr_tb = pd.read_csv(cwd_slash('lr_scan.csv'), index_col=0)
    lr_tb['loss_chart'] = chart(lr_tb['loss'], 30)
    max_rows_bk = pd.options.display.max_rows
    pd.options.display.max_rows = config['lr_scan_n_epochs']
    print(lr_tb)
    pd.options.display.max_rows = max_rows_bk

    return float(input('Set LR to: '))


def gen_np_macro_f1_printing_details(config):
    cwd_slash = gen_cwd_slash(config)
    log_folder = cwd_slash('metric_details')
    os.makedirs(log_folder, exist_ok=True)

    def np_macro_f1_printing_details(y_true, y_pred, epoch, logs):
        macro_f1_score, details = np_macro_f1(y_true, y_pred, config, return_details=True)
        save_path = pjoin(log_folder, f"{logs['end_time']}__{epoch}__{macro_f1_score}.csv")
        formatted_details = format_macro_f1_details(details, config)
        print(formatted_details)
        formatted_details.to_csv(save_path)
        debug(f'saved logs to {save_path}')
        return macro_f1_score

    return np_macro_f1_printing_details


def train(config):
    print_config(config)
    cwd_slash = gen_cwd_slash(config)

    os.makedirs(config['_cwd'], exist_ok=True)

    if config['cuda_visible_devices'] is not None:
        debug(f"Using GPU: {config['cuda_visible_devices']}")
        os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_visible_devices']
    else:
        avail_gpu = str(GPUtil.getFirstAvailable()[0])
        debug(f"Selecting the first available GPU: {avail_gpu}")
        os.environ['CUDA_VISIBLE_DEVICES'] = avail_gpu

    if os.path.exists(cwd_slash('history.csv')):
        try:
            last_history = pd.read_csv(cwd_slash('history.csv'))
            last_best = last_history[config['metric']].values.max()
            debug(f"found last_best = {last_best} in {cwd_slash('history.csv')}")
        except Exception:
            debug(f"failed parsing {cwd_slash('history.csv')}, will set last_best = None")
            last_best = None
    else:
        last_best = None

    train_anno = pd.read_csv(config['path_to_train_anno_cache'], index_col=0)
    valid_anno = pd.read_csv(config['path_to_valid_anno_cache'], index_col=0)

    train_windowed_anno = pd.read_csv(config['path_to_train_windowed_anno'], index_col=0)
    train_windowed_anno = train_windowed_anno.join(train_anno, how='right', on='source_img_id')
    train_windowed_anno.to_csv(cwd_slash('train_windowed_anno.csv'))
    valid_windowed_anno = pd.read_csv(config['path_to_valid_windowed_anno'], index_col=0)
    valid_windowed_anno = valid_windowed_anno.join(valid_anno, how='right', on='source_img_id')
    valid_windowed_anno.to_csv(cwd_slash('valid_windowed_anno.csv'))

    train_anno = train_windowed_anno
    train_anno['Target'] = train_anno['corrected_target']
    valid_anno = valid_windowed_anno
    valid_anno['Target'] = valid_anno['corrected_target']

    debug(f'len(valid_anno) = {len(valid_anno)}')

    # create train and valid datagens
    train_generator = data_gen_from_anno_gen(
        gen_even_batches(train_anno, config),
        config,
        folder=config['path_to_train'],
        extension=config['img_extension'],
    )
    debug(f"train_generator = {train_generator}")
    # train_generator = create_generator_from_anno(train_anno, config['path_to_train'], config)

    if config['n_batches_preview'] > 0:
        debug('preview_generator ...')
        preview_generator(
            train_generator,
            config,
            filename_prefix='train_generator',
            n_batches=config['n_batches_preview'],
        )
        debug('preview_generator done')

    # train_generator = create_generator_from_anno(
    #     train_anno,
    #     config=config,
    #     do_augment=config['augmentation_at_training'],
    #     do_shuffle=True,
    # )

    valid_generator = data_gen_from_anno_gen(
        gen_even_batches(valid_anno, config),
        config,
        folder=config['path_to_valid'],
        extension=config['img_extension'],
    )
    debug(f"valid_generator = {valid_generator}")

    # valid_generator = create_generator_from_anno(valid_anno, config['path_to_valid'], config)
    if config['n_batches_preview'] > 0:
        debug('preview_generator ...')
        preview_generator(
            valid_generator,
            config,
            filename_prefix='valid_generator',
            n_batches=config['n_batches_preview'],
        )
        debug('preview_generator done')
    # valid_generator = create_generator_from_anno(
    #     valid_anno,
    #     config=config,
    # )

    # model = create_model(config)
    model_specs_path = cwd_slash('model.yaml')
    if not os.path.exists(model_specs_path):
        input_shape = config['_input_shape']

        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            # weights=None,
            input_shape=(input_shape[0], input_shape[1], 3),
        )

        input_tensor = Input(shape=input_shape)
        x = BatchNormalization()(input_tensor)
        # x = Conv2D(3, kernel_size=(1, 1), activation='relu')(x)  # converting 4 channels into 3
        # TODO: seems like the new model tends to over-fit: add another Dropout + Dense layers?
        x = base_model(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)
        # TODO: concat GlobalAveragePooling2D with GlobalMaxPooling2D?
        # x = Dropout(0.5)(x)
        # TODO: some say sigmoid should not be used?
        # TODO: COUNTINUE: https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-448-public-lb
        # TODO: add double dense layers? 2048 -> 1024 -> 28?
        x = Dense(config['_n_classes'], activation='sigmoid')(x)
        model = Model(input_tensor, x)
        with open(model_specs_path, 'w') as f:
            f.write(model.to_yaml())
    else:
        with open(model_specs_path, 'r') as f:
            model = model_from_yaml(f.read())

    if config['do_lr_scanning']:
        # scan through LRs to find the best one
        starting_lr = lr_scanning(model, train_generator, config, begin_lr=1e-9, end_lr=1e-1)
    else:
        starting_lr = config['starting_lr']

    if os.path.isfile(cwd_slash('latest.weights')):
        debug(f"Load weights from {cwd_slash('latest.weights')}")
        model.load_weights(cwd_slash('latest.weights'))

    # if config['pretraining_n_epochs'] > 0 and not os.path.exists(cwd_slash('history.csv')):
    #     # TODO: Need a better trainability handling for the base_model
    #     base_model.trainable = False
    #     # TODO: we should probably use "focal loss" to migitate sample imbalance https://arxiv.org/pdf/1708.02002.pdf
    #     # NOTE: Do *not* use macro_f1_loss as the loss function (bad mathematical property)
    #     model.compile(
    #         loss='binary_crossentropy',
    #         # loss=focal_loss,
    #         optimizer=Adam(lr=1e-03),
    #         metrics=['accuracy', gen_macro_f1_metric(config)],
    #     )
    #     model.summary()
    #     info(f"Pretraining is enabled! (for {config['pretraining_n_epochs']} epochs)")
    #     model.fit_generator(
    #         train_generator,
    #         steps_per_epoch=config['steps_per_epoch'],
    #         # validation_data=valid_generator,
    #         # validation_steps=n_batches_in_valid,
    #         max_queue_size=config['max_queue_size'],
    #         epochs=config['pretraining_n_epochs'],
    #         verbose=config['verbose'],
    #         callbacks=[
    #             TimeLogger(),
    #             # MetricOnAll(
    #             #     metric_name=config['metric'],
    #             #     validation_data=valid_generator,
    #             #     validation_steps=10,
    #             #     batch_size=config['batch_size'],
    #             #     metric_fn=gen_np_macro_f1_printing_details(config),
    #             #     # pred_fn=lambda x: np.where(x >= config['score_threshold'], 1, 0),
    #             #     verbose=config['verbose'],
    #             # ),
    #             # TensorBoard(log_dir=config['_cwd']),
    #             # CSVLogger(cwd_slash('history.csv'), append=True),
    #         ],
    #     )

    #     # # TODO: EXP: how much difference is there between releasing all layers or not?
    #     # # TODO: EXP: show some examples of the "most wrong" predictions in the validation dataset.

    #     base_model.trainable = True

    model.compile(
        loss='binary_crossentropy',
        # loss=focal_loss,
        optimizer=Adam(lr=starting_lr),
        metrics=['accuracy', gen_macro_f1_metric(config)],
    )
    model.summary()
    info(f"Starting training (for {config['n_epochs']} epochs with starting_lr = {config['starting_lr']})")
    try:
        model.fit_generator(
            train_generator,
            steps_per_epoch=config['steps_per_epoch'],
            epochs=config['n_epochs'],
            # validation_data=valid_generator,
            # validation_steps=n_batches_in_valid,
            verbose=config['verbose'],
            max_queue_size=config['max_queue_size'],
            callbacks=[
                TimeLogger(),
                MetricOnAll(
                    metric_name=config['metric'],
                    validation_data=valid_generator,
                    validation_steps=10,
                    batch_size=config['batch_size'],
                    metric_fn=gen_np_macro_f1_printing_details(config),
                    # pred_fn=lambda x: np.where(x >= config['score_threshold'], 1, 0),
                    verbose=config['verbose'],
                ),
                # TensorBoard(log_dir=config['_cwd'], update_freq=10000),
                ModelCheckpoint(
                    cwd_slash('latest.weights'),
                    monitor=config['metric'],
                    verbose=config['verbose'],
                    save_best_only=False,
                    mode='max',
                    save_weights_only=True,
                ),
                ModelCheckpoint(
                    cwd_slash('best.weights'),
                    monitor=config['metric'],
                    verbose=config['verbose'],
                    save_best_only=True,
                    mode='max',
                    save_weights_only=True,
                    last_best=last_best,
                ),
                # LearningRateScheduler(
                #     gen_cosine_annealing(period=10, upper=starting_lr, lower=1e-5),
                #     verbose=config['verbose'],
                # ),
                ReduceLROnPlateau(
                    monitor='loss',
                    # monitor=config['metric'],
                    factor=0.3,
                    patience=1,
                    verbose=config['verbose'],
                    mode='min',
                    # mode='max',
                    min_delta=0.0001,
                ),
                # EarlyStopping(
                #     monitor=config['metric'],
                #     mode="max",
                #     patience=config['early_stopping_patience'],
                # ),
                CSVLogger(cwd_slash('history.csv'), append=True),
            ],
        )
    except KeyboardInterrupt:
        pass

    after_training(config)

    return {'id_': config['class_ids']}
