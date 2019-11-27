import warnings
from datetime import datetime

import numpy as np
from .utils import debug
from keras.callbacks import Callback


class MetricOnAll(Callback):

    def __init__(
        self,
        metric_name,
        validation_data,
        validation_steps,
        batch_size,
        metric_fn,
        pred_fn=lambda x: x,
        verbose=1,
        **kwargs,
    ):
        super().__init__()

        self.metric_name = metric_name
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.batch_size = batch_size
        self.metric_fn = metric_fn
        self.pred_fn = pred_fn
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        scores_predicted_list = []
        targets_list = []
        for i in range(self.validation_steps):
            imgs, targets = next(self.validation_data)
            scores_predicted_list.append(self.model.predict(imgs, batch_size=self.batch_size))
            targets_list.append(targets)
        y_pred = self.pred_fn(np.concatenate(scores_predicted_list))
        y_true = np.concatenate(targets_list)
        metric_result = self.metric_fn(y_true, y_pred, epoch, logs)
        if self.verbose > 0:
            print(f'{self.metric_name} (on {y_pred.shape[0]} samples) = {metric_result}')

        logs[self.metric_name] = metric_result


class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(
        self,
        filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        last_best=None,
        mode='auto',
        period=1
    ):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, ' 'fallback to auto mode.' % (mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            if last_best is not None:
                self.best = last_best
            else:
                self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            if last_best is not None:
                self.best = last_best
            else:
                self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                if last_best is not None:
                    self.best = last_best
                else:
                    self.best = -np.Inf
            else:
                self.monitor_op = np.less
                if last_best is not None:
                    self.best = last_best
                else:
                    self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        'Can save best model only with %s available, '
                        'skipping.' % (self.monitor), RuntimeWarning
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                '\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                ' saving model to %s' % (epoch + 1, self.monitor, self.best, current, filepath)
                            )
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' % (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class TimeLogger(Callback):

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_begin_time = datetime.now()

    def on_epoch_end(self, batch, logs={}):
        logs['begin_time'] = self.epoch_begin_time.isoformat()
        logs['end_time'] = datetime.now().isoformat()
        logs['duration'] = str(datetime.now() - self.epoch_begin_time)
