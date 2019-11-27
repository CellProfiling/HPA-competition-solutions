import json
import math
import os
from datetime import timedelta

import colors
import GPUtil
import ignite
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (Events, create_supervised_evaluator, create_supervised_trainer)
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, Metric
from torch import nn
from tqdm import tqdm

from .combined_net import CombinedNet
from .handlers import MacroF1, RunningAverage, CollectedPrediction

# from .model import MyMobileNetV2 as MyModel
from .model import MyXception as MyModel
from ..utils import (
    ChunkIter,
    data_gen_from_anno_gen,
    debug,
    info,
    banner,
    format_macro_f1_details,
    gen_cwd_slash,
    gen_even_batches,
    chunk,
    batching_row_gen,
    randomize_and_loop,
    preview_generator,
    class_ids_to_label,
    load_img,
    RedisCached,
    chunk_df,
)


def numpy_to_pytorch(iterable):
    for x, y in iterable:
        yield (torch.from_numpy(np.transpose(x, axes=[0, 3, 1, 2])), torch.from_numpy(y).float())


def train(config):
    cwd_slash = gen_cwd_slash(config)

    os.makedirs(config['_cwd'], exist_ok=True)

    if config['cuda_visible_devices'] is not None:
        debug(f"Using GPU: {config['cuda_visible_devices']}")
        os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_visible_devices']
    else:
        avail_gpu = str(GPUtil.getFirstAvailable()[0])
        debug(f"Selecting the first available GPU: {avail_gpu}")
        os.environ['CUDA_VISIBLE_DEVICES'] = avail_gpu

    train_windowed_anno = pd.read_csv(config['path_to_train_windowed_anno_cache'], index_col=0)
    valid_windowed_anno = pd.read_csv(config['path_to_valid_windowed_anno_cache'], index_col=0)

    train_balanced_generator = data_gen_from_anno_gen(
        gen_even_batches(
            train_windowed_anno,
            config,
            target_col='corrected_target',
        ),
        config,
        target_col='corrected_target',
        do_augment=True,
    )

    if config['n_batches_preview'] > 0:
        debug('preview_generator ...')
        preview_generator(
            train_balanced_generator,
            config,
            filename_prefix=f"train_balanced_generator_{'_'.join([str(x) for x in config['class_ids']])}",
            n_batches=config['n_batches_preview'],
        )
        debug('preview_generator done')

    train_generator = data_gen_from_anno_gen(
        batching_row_gen(randomize_and_loop(train_windowed_anno), config['batch_size']),
        config,
        target_col='corrected_target',
        do_augment=True,
    )

    if config['n_batches_preview'] > 0:
        debug('preview_generator ...')
        preview_generator(
            train_generator,
            config,
            filename_prefix=f"train_generator_{'_'.join([str(x) for x in config['class_ids']])}",
            n_batches=config['n_batches_preview'],
        )
        debug('preview_generator done')

    valid_balanced_generator = data_gen_from_anno_gen(
        gen_even_batches(
            valid_windowed_anno,
            config,
            target_col='corrected_target',
        ),
        config,
        target_col='corrected_target',
        do_augment=False,
    )

    if config['n_batches_preview'] > 0:
        debug('preview_generator ...')
        preview_generator(
            valid_balanced_generator,
            config,
            filename_prefix=f"valid_balanced_generator_{'_'.join([str(x) for x in config['class_ids']])}",
            n_batches=config['n_batches_preview'],
        )
        debug('preview_generator done')

    device = 'cuda'
    log_interval = 1

    train_balanced_generator = numpy_to_pytorch(train_balanced_generator)
    train_balanced_loader = ChunkIter(train_balanced_generator, config['steps_per_epoch'])
    train_generator = numpy_to_pytorch(train_generator)
    train_loader = ChunkIter(train_generator, config['steps_per_epoch'])
    valid_balanced_generator = numpy_to_pytorch(valid_balanced_generator)
    val_loader = ChunkIter(valid_balanced_generator, config['steps_per_epoch_for_valid'])

    def debug_hook(module, input_, output):
        debug(f"input_ = {input_}")
        debug(f"output = {output}")

    model = MyModel(config)
    model.to(device)

    path_to_model_checkpoint = cwd_slash('model_best_ravg_loss.pth')
    if os.path.exists(path_to_model_checkpoint):
        debug(f"loading model checkpoint from {path_to_model_checkpoint}")
        model_state_dict = torch.load(path_to_model_checkpoint)
        model.load_state_dict(model_state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['starting_lr'])

    path_to_optimizer_checkpoint = cwd_slash('optimizer_best_ravg_loss.pth')
    if os.path.exists(path_to_optimizer_checkpoint):
        debug(f"loading optimizer checkpoint from {path_to_optimizer_checkpoint}")
        optimizer_state_dict = torch.load(path_to_optimizer_checkpoint)
        optimizer.load_state_dict(optimizer_state_dict)

    trainer = create_supervised_trainer(model, optimizer, F.binary_cross_entropy, device=device)
    RunningAverage(alpha=0.99).attach(trainer, 'ravg_loss')
    epoch_timer = Timer().attach(trainer, start=Events.EPOCH_STARTED)

    metrics = {
        'acc': Accuracy(),
        'val_macro_f1': MacroF1(),
        # 'nll': Loss(F.binary_cross_entropy),
    }
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    pbar = tqdm(initial=0, leave=False, total=config['steps_per_epoch'], mininterval=0.1)

    @trainer.on(Events.STARTED)
    def started_handler(engine):
        info("started_handler()")
        engine.state.last_ravg_loss = math.inf
        engine.state.best_val_macro_f1 = 0
        engine.state.best_ravg_loss = math.inf
        engine.state.lr = config['starting_lr']
        engine.state.n_restarts = 0
        engine.state.ravg_loss_improved = False
        engine.state.val_macro_f1_improved = False

        def format_log_header(fields):
            output_groups = []
            output_group = []
            for field in fields:
                name = field.get('name')
                if name is None:
                    output_groups.append(output_group)
                    output_group = []
                    continue

                display_str = name

                width = field.get('width')
                if type(width) is int:
                    if int < 0:
                        display_str.ljust(-width)
                    else:
                        display_str.rjust(width)

                output_group.append(display_str)

            output_groups.append(output_group)

            return ' | '.join([' '.join(g) for g in output_groups])

    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_completed_handler(engine):
        iter = (engine.state.iteration - 1) % config['steps_per_epoch'] + 1

        if iter % log_interval == 0:
            pbar.set_description_str(
                ' | '.join(
                    [
                        # "class_ids " + str(config['class_ids']).rjust(4),
                        "fold " + str(config['i_fold']),
                        "epoch " + str(engine.state.epoch).rjust(2),
                        "ravg_loss " + f"{trainer.state.metrics['ravg_loss']:.6f}",
                        "loss " + f"{engine.state.output:.4f}",
                    ]
                )
            )
            pbar.update(log_interval)

    max_n_restarts = 10

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_completed_handler(engine):
        pbar.refresh()
        events = []
        # evaluator.run(train_balanced_loader)
        evaluator.run(val_loader)
        val_metrics = evaluator.state.metrics
        val_macro_f1 = val_metrics['val_macro_f1']['score'].item()
        score_details = val_metrics['val_macro_f1']['details']
        precisions = score_details['precision']
        recalls = score_details['recall']
        # log_record = {
        #     "epoch": engine.state.epoch,
        #     "ravg_loss": engine.state.metrics['ravg_loss'],
        #     "val_avg_acc": val_metrics['acc'],
        #     "val_macro_f1": val_macro_f1,
        #     "epoch_time": epoch_timer.value(),
        #     "engine.state.lr": engine.state.lr,
        # }
        os.makedirs(cwd_slash('macro_f1_details'), exist_ok=True)

        if engine.state.last_ravg_loss - engine.state.metrics['ravg_loss'] < 0.025 * engine.state.last_ravg_loss:
            engine.state.ravg_loss_improved = False
            engine.state.lr *= 0.3
            if engine.state.lr < 5e-6:
                model_state_save_path = cwd_slash(f"model_restart_{engine.state.n_restarts}.pth")
                torch.save(model.state_dict(), model_state_save_path)
                debug(f"saved {model_state_save_path}")

                model_softlink_path = cwd_slash(f"model.pth")
                debug(f"overwriting soft link {model_softlink_path}  -->  {model_state_save_path}")
                if os.path.islink(model_softlink_path):
                    os.unlink(model_softlink_path)
                os.symlink(
                    os.path.relpath(model_state_save_path, cwd_slash()),
                    model_softlink_path,
                    target_is_directory=True,
                )

                optimizer_state_save_path = cwd_slash(f"optimizer_restart_{engine.state.n_restarts}.pth")
                torch.save(optimizer.state_dict(), optimizer_state_save_path)
                debug(f"saved {optimizer_state_save_path}")

                optimizer_softlink_path = cwd_slash(f"optimizer.pth")
                debug(f"overwriting soft link {optimizer_softlink_path}  -->  {optimizer_state_save_path}")
                if os.path.islink(optimizer_softlink_path):
                    os.unlink(optimizer_softlink_path)
                os.symlink(
                    os.path.relpath(optimizer_state_save_path, cwd_slash()),
                    optimizer_softlink_path,
                    target_is_directory=True,
                )

                engine.state.n_restarts += 1
                if engine.state.n_restarts > max_n_restarts:
                    engine.terminate()
                    events.append(f"max restarts reached")
                else:
                    engine.state.last_ravg_loss = math.inf
                    engine.state.lr = config['starting_lr']
                    events.append(f"lr reset to {engine.state.lr:.1e}")
            else:
                events.append(f"lr -> {engine.state.lr:.1e}")
            for g in optimizer.param_groups:
                g['lr'] = engine.state.lr
        else:
            engine.state.ravg_loss_improved = True

        if engine.state.metrics['ravg_loss'] < engine.state.best_ravg_loss:
            engine.state.best_ravg_loss = engine.state.metrics['ravg_loss']
            debug(f"saved {cwd_slash('model_best_ravg_loss.pth')}")
            debug(f"saved {cwd_slash('optimizer_best_ravg_loss.pth')}")
            torch.save(model.state_dict(), cwd_slash('model_best_ravg_loss.pth'))
            torch.save(optimizer.state_dict(), cwd_slash('optimizer_best_ravg_loss.pth'))

        if val_macro_f1 > engine.state.best_val_macro_f1:
            engine.state.val_macro_f1_improved = True
            engine.state.best_val_macro_f1 = val_macro_f1
            debug(f"saved {cwd_slash('model_best_val_f1.pth')}")
            debug(f"saved {cwd_slash('optimizer_best_val_f1.pth')}")
            torch.save(model.state_dict(), cwd_slash('model_best_val_f1.pth'))
            torch.save(optimizer.state_dict(), cwd_slash('optimizer_best_val_f1.pth'))
        else:
            engine.state.val_macro_f1_improved = False

        log_record = [
            # {
            #     'name': 'class_labels',
            #     'value': class_ids_to_label(config['class_ids'], config),
            #     'width': -32,
            # },
            # {
            #     # ------------------
            # },
            {
                'name': 'fold',
                'value': config['i_fold'],
                'width': 1,
            },
            {
                # ------------------
            },
            {
                'name': 'epoch',
                'value': engine.state.epoch,
                'width': 3,
            },
            {
                # ------------------
            },
            {
                'name': 'ravg_loss',
                'value': engine.state.metrics['ravg_loss'],
                'display': "{:.6f}",
                'width': -9,
                'color': 'yellow' if engine.state.ravg_loss_improved else None,
            },
            {
                'name': 'val_avg_acc',
                'value': val_metrics['acc'],
                'display': "{:.4f}",
                'width': 6,
            },
            {
                # ------------------
            },
            {
                'name': 'val_macro_f1',
                'value': val_macro_f1,
                'display': "{:.6f}",
                'width': -9,
                'color': 'blue' if engine.state.val_macro_f1_improved else None,
            },
            {
                'name': 'precision',
                'value': float(precisions[0]),
                'display': "{:.4f}",
                'width': 6,
            },
            {
                'name': 'recall',
                'value': float(recalls[0]),
                'display': "{:.4f}",
                'width': 6,
            },
            {
                # ------------------
            },
            {
                'name': "epoch_time",
                'value': epoch_timer.value(),
                'display': lambda x: timedelta(seconds=x),
                'width': 15,
            },
            {
                # ------------------
            },
            {
                'name': "lr",
                'value': engine.state.lr,
                'display': "{:.1e}",
                'width': 7,
            },
            {
                # ------------------
            },
            {
                'name': "cache",
                'value': load_img.cache_info(),
                'width': 18,
            },
            {
                # ------------------
            },
            {
                'name': "events",
                'value': '; '.join(events),
            },
        ]

        def format_log_record(fields):
            output_groups = []
            output_group = []
            for field in fields:
                name = field.get('name')
                if name is None:
                    output_groups.append(output_group)
                    output_group = []
                    continue

                value = field.get('value')
                display = field.get('display')
                if type(display) is str:
                    display_str = display.format(value)
                elif callable(display):
                    display_str = str(display(value))
                else:
                    display_str = str(value)

                width = field.get('width')
                if type(width) is int:
                    if width < 0:
                        display_str = display_str.ljust(-width)
                    else:
                        display_str = display_str.rjust(width)

                color = field.get('color')
                if type(color) is str:
                    display_str = colors.color(display_str, fg=color)

                output_group.append(display_str)

            output_groups.append(output_group)

            return ' | '.join([' '.join(g) for g in output_groups])

        with open(cwd_slash('log.json'), 'a') as f:
            obj = {x['name']: x['value'] for x in log_record if type(x.get('name')) is str}
            json.dump(obj, f)
            f.write('\n')

        with open(cwd_slash('displayed_log.txt'), 'a') as f:
            f.write(format_log_record(log_record))
            f.write('\n')

        tqdm.write(format_log_record(log_record))

        load_img.reset_cache_info()
        macro_f1_df = format_macro_f1_details(val_metrics['val_macro_f1']['details'], config)
        macro_f1_df.to_csv(cwd_slash('macro_f1_details', f"epoch{engine.state.epoch:03d}_{val_macro_f1}.csv"))
        tqdm.write(repr(macro_f1_df))

        engine.state.last_ravg_loss = engine.state.metrics['ravg_loss']
        pbar.n = pbar.last_print_n = 0
        pbar.refresh()

    # banner('start training train_balanced_loader')
    trainer.run(train_balanced_loader, max_epochs=config['n_epochs'])
    # trainer.should_terminate = False
    # max_n_restarts = 10
    # banner('start training train_loader')
    # trainer.run(train_loader, max_epochs=config['n_epochs'])

    return {'id_': config['class_ids']}


def predict(config, anno, path_to_model_checkpoint, target_col=None, save_numpy_to=None, save_csv_to=None):
    cwd_slash = gen_cwd_slash(config)
    device = 'cuda'

    model = MyModel(config, pretrained=None, no_fc=True)
    model.to(device)
    state_dict = torch.load(path_to_model_checkpoint)
    model.load_state_dict(state_dict)

    model = nn.DataParallel(model)

    metrics = {}
    metrics['collected_prediction'] = CollectedPrediction()
    if type(target_col) is str:
        macro_f1_metric = MacroF1()
        metrics['macro_f1'] = macro_f1_metric

    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    valid_generator = numpy_to_pytorch(
        data_gen_from_anno_gen(
            chunk_df(anno, config['predict_batch_size'] * len(model.device_ids)),
            config,
            target_col=target_col,
            do_augment=False,
        )
    )
    n_iters = math.ceil(len(anno) / (config['predict_batch_size'] * len(model.device_ids)))

    pbar = tqdm(initial=0, leave=False, total=n_iters, mininterval=0.1)
    log_interval = 10

    @evaluator.on(Events.ITERATION_COMPLETED)
    def iteration_completed_handler(engine):
        if type(target_col) is str:
            if engine.state.iteration % log_interval == 0:
                macro_f1 = macro_f1_metric.compute()
                tqdm.write(str(format_macro_f1_details(macro_f1['details'], config)))
                tqdm.write(str(macro_f1['score']))
        pbar.update()

    evaluator.run(valid_generator)
    pbar.close()

    prediction = evaluator.state.metrics['collected_prediction']
    prediction = prediction.cpu().numpy()
    if save_numpy_to is not None:
        np.save(cwd_slash(save_numpy_to), prediction)
        debug(f"saved to {cwd_slash(save_numpy_to)}")

    anno_predicted = pd.concat([anno.reset_index(), pd.DataFrame(prediction)], axis=1)
    if save_csv_to is not None:
        anno_predicted.to_csv(cwd_slash(save_csv_to), index=False)
        debug(f"saved to {cwd_slash(save_csv_to)}")


# def combined_predict(config, anno, save_numpy_to=None, save_csv_to=None):
#     cwd_slash = gen_cwd_slash(config)

#     device = 'cuda'

#     state_dicts = []
#     for class_id in config['class_ids']:
#         state_dict = torch.load(cwd_slash(str(class_id), 'model_state_dict.pt'), map_location=device)
#         state_dicts.append(state_dict)

#     def get_sub_model():
#         return nn.Sequential(
#             MobileNetV2(n_input_channels=config['n_channels'], n_class=1, input_size=config['size']),
#             nn.Sigmoid(),
#         )

#     model = nn.DataParallel(CombinedNet(get_sub_model, state_dicts))
#     debug(f"model.device_ids = {model.device_ids}")
#     macro_f1_metric = MacroF1(threshold=0.9)
#     metrics = {
#         'macro_f1': macro_f1_metric,
#         'collected_prediction': CollectedPrediction(),
#     }
#     evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

#     valid_generator = numpy_to_pytorch(
#         data_gen_from_anno_gen(
#             chunk_df(anno, config['predict_batch_size'] * len(model.device_ids)),
#             config,
#             target_col='corrected_target',
#             do_augment=False,
#         )
#     )
#     n_iters = math.ceil(len(anno) / (config['predict_batch_size'] * len(model.device_ids)))

#     pbar = tqdm(initial=0, leave=False, total=n_iters, mininterval=0.1)
#     log_interval = 10

#     @evaluator.on(Events.ITERATION_COMPLETED)
#     def iteration_completed_handler(engine):
#         if engine.state.iteration % log_interval == 0:
#             macro_f1 = macro_f1_metric.compute()
#             tqdm.write(str(format_macro_f1_details(macro_f1['details'], config)))
#             tqdm.write(str(macro_f1['score']))
#         pbar.update()

#     evaluator.run(valid_generator)
#     pbar.close()

#     prediction = evaluator.state.metrics['collected_prediction']
#     prediction = prediction.cpu().numpy()
#     if save_numpy_to is not None:
#         np.save(cwd_slash(save_numpy_to), prediction)
#         debug(f"saved to {cwd_slash(save_numpy_to)}")

#     anno_predicted = pd.concat([anno.reset_index(), pd.DataFrame(prediction)], axis=1)
#     if save_csv_to is not None:
#         anno_predicted.to_csv(cwd_slash(save_csv_to), index=False)
#         debug(f"saved to {cwd_slash(save_csv_to)}")

#     return {}
