import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class F1Metric(Callback):

    def __init__(self,val_gen, steps, batch_size, num_classes, logit_scale=False):
        super().__init__()
        self.val_gen = val_gen
        self.f1_val_steps = steps
        self.f1_val_bsize = batch_size
        self.nclasses = num_classes
        self.rares = np.array([8, 9, 10, 15, 27])
        self.do_digmoid = logit_scale

    def on_train_begin(self, logs={}):
        self.val_f1s = []


    def on_epoch_end(self, epoch, logs={}):

        preds = np.zeros(shape=(self.f1_val_steps,self.f1_val_bsize,self.nclasses))
        y_trues = np.zeros(shape=(self.f1_val_steps,self.f1_val_bsize,self.nclasses))
        for s in range(self.f1_val_steps):
            x, y = next(self.val_gen)
            if x.shape[0] == self.f1_val_bsize:
                pred = np.asarray((self.model.predict(x)))
                if self.do_digmoid:
                    pred = 1/(1+np.exp(-pred))
                pred = pred > 0.5
                preds[s] = pred
                y_trues[s] = y
        _val_f1 = f1_score(np.reshape(y_trues, (-1, self.nclasses)), np.reshape(preds, (-1, self.nclasses)), average='macro')
        logs['val_f1_all'] = _val_f1
        print("val_f1_all: {}".format(_val_f1))
        f1s = [f1_score(np.reshape(y_trues, (-1, self.nclasses))[:,i], np.reshape(preds, (-1, self.nclasses))[:,i]) for i in self.rares]
        _val_f1_rares = np.mean(f1s)
        logs['val_f1_rares'] = _val_f1_rares
        print("val_f1_rares: {}".format(_val_f1_rares))
        return

class F1MetricField(Callback):

    def __init__(self,val_gen, steps, batch_size, num_classes,gpct = 95.):
        super().__init__()
        self.val_gen = val_gen
        self.f1_val_steps = steps
        self.f1_val_bsize = batch_size
        self.nclasses = num_classes
        self.gpct = gpct

    def on_train_begin(self, logs={}):
        self.val_f1s = []


    def on_epoch_end(self, epoch, logs={}):

        preds = np.zeros(shape=(self.f1_val_steps,self.f1_val_bsize,32,32,self.nclasses))
        y_trues = np.zeros(shape=(self.f1_val_steps,self.f1_val_bsize,self.nclasses))
        for s in range(self.f1_val_steps):
            x, y = next(self.val_gen)
            if x.shape[0] == self.f1_val_bsize:
                pred = np.asarray((self.model.predict(x)))

                preds[s] = pred
                #vts = np.vstack(y_trues)
                vts_max = np.max(y, axis=(1, 2))
                vts = (vts_max > 0).astype(float)

                y_trues[s] = vts


        vps = np.vstack(preds)
        vts = np.vstack(y_trues)
        vpsp = np.percentile(vps, self.gpct, axis=(1, 2))
        thresholds = np.linspace(0, 1, 101)
        scores = np.array([f1_score(vts, np.int32(vpsp > t),
                                    average='macro') for t in thresholds])
        threshold_best_index = np.argmax(scores)
        vf1 = scores[threshold_best_index]

        #pred = pred > 0.5
        #_val_f1 = f1_score(np.reshape(y_trues, (-1, self.nclasses)), np.reshape(preds, (-1, self.nclasses)), average='macro')
        logs['val_f1_all'] = vf1
        print("val_f1_all at {:.3}: {}".format(thresholds[threshold_best_index],vf1))
        #f1s = [f1_score(np.reshape(y_trues, (-1, self.nclasses))[:,i], np.reshape(preds, (-1, self.nclasses))[:,i]) for i in self.rares]
        #_val_f1_rares = np.mean(f1s)
        #logs['val_f1_rares'] = _val_f1_rares
        #print("val_f1_rares: {}".format(_val_f1_rares))
        return


import os
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler



class SnapshotModelCheckpoint(Callback):
    """Callback that saves the snapshot weights of the model.
    Saves the model weights on certain epochs (which can be considered the
    snapshot of the model at that epoch).
    Should be used with the cosine annealing learning rate schedule to save
    the weight just before learning rate is sharply increased.
    # Arguments:
        nb_epochs: total number of epochs that the model will be trained for.
        nb_snapshots: number of times the weights of the model will be saved.
        fn_prefix: prefix for the filename of the weights.
    """

    def __init__(self, nb_epochs, nb_snapshots, fn_prefix='Model'):
        super(SnapshotModelCheckpoint, self).__init__()

        self.check = nb_epochs // nb_snapshots
        self.fn_prefix = fn_prefix

    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and (epoch + 1) % self.check == 0:
            filepath = self.fn_prefix + '-%d.h5' % ((epoch + 1) // self.check)
            self.model.save_weights(filepath, overwrite=True)
            # print("Saved snapshot at weights/%s_%d.h5" % (self.fn_prefix, epoch))


class SnapshotCallbackBuilder:
    """Callback builder for snapshot ensemble training of a model.
    From the paper "Snapshot Ensembles: Train 1, Get M For Free" (https://openreview.net/pdf?id=BJYwwY9ll)
    Creates a list of callbacks, which are provided when training a model
    so as to save the model weights at certain epochs, and then sharply
    increase the learning rate.
    """

    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1,model_path=None,model_prefix='model',monitor='val_competition_metric'):
        """
        Initialize a snapshot callback builder.
        # Arguments:
            nb_epochs: total number of epochs that the model will be trained for.
            nb_snapshots: number of times the weights of the model will be saved.
            init_lr: initial learning rate
        """
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.model_path=model_path
        self.model_prefix = model_prefix
        self.monitor = monitor

    def get_callbacks(self):
        """
        Creates a list of callbacks that can be used during training to create a
        snapshot ensemble of the model.
        Args:
            model_prefix: prefix for the filename of the weights.
        Returns: list of 3 callbacks [ModelCheckpoint, LearningRateScheduler,
                 SnapshotModelCheckpoint] which can be provided to the 'fit' function
        """
        if not os.path.exists(self.model_path + 'weights/'):
            os.makedirs(self.model_path + 'weights/')

        callback_list = [ModelCheckpoint(self.model_path + 'weights/%s-best.h5' % self.model_prefix, monitor=self.monitor,mode='max',
                                         save_best_only=True, save_weights_only=True,verbose=True),
                         LearningRateScheduler(schedule=self._cosine_anneal_schedule,verbose=True),
                         SnapshotModelCheckpoint(self.T, self.M, fn_prefix=self.model_path + 'weights/%s' % self.model_prefix)]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)

import keras
from keras.callbacks import Callback
from keras import backend as K
import math
import random
import numpy as np
import matplotlib.pyplot as plt

if K.backend() == 'theano':
    if K.image_data_format() == 'channels_last':
        K.set_image_data_format('channels_first')
else:
    if K.image_data_format() == 'channels_first':
        K.set_image_data_format('channels_last')

class LR_Updater(Callback):
    '''This callback is utilized to log learning rates every iteration (batch cycle)
    it is not meant to be directly used as a callback but extended by other callbacks
    ie. LR_Cycle
    '''

    def __init__(self, iterations, epochs=1):
        super().__init__()
        '''
        iterations = dataset size / batch size
        epochs = pass through full training dataset
        '''
        self.epoch_iterations = iterations
        self.trn_iterations = 0.
        self.history = {}

    def setRate(self):
        return K.get_value(self.model.optimizer.lr)

    def on_train_begin(self, logs={}):
        self.trn_iterations = 0.
        logs = logs or {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        K.set_value(self.model.optimizer.lr, self.setRate())
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def plot_lr(self):
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.history['iterations'], self.history['lr'])

    def plot(self, n_skip=10):
        plt.xlabel("learning rate (log scale)")
        plt.ylabel("loss")
        plt.plot(self.history['lr'][n_skip:-5], self.history['loss'][n_skip:-5])
        plt.xscale('log')

class LR_Find(LR_Updater):
    '''This callback is utilized to determine the optimal lr to be used
    it is based on this pytorch implementation https://github.com/fastai/fastai/blob/master/fastai/learner.py
    and adopted from this keras implementation https://github.com/bckenstler/CLR
    it loosely implements methods described in the paper https://arxiv.org/pdf/1506.01186.pdf
    '''

    def __init__(self, iterations, epochs=1, min_lr=1e-05, max_lr=10, jump=6):
        '''
        iterations = dataset size / batch size
        epochs should always be 1
        min_lr is the starting learning rate
        max_lr is the upper bound of the learning rate
        jump is the x-fold loss increase that will cause training to stop (defaults to 6)
        '''
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_mult = (max_lr / min_lr) ** (1 / iterations)
        self.jump = jump
        super().__init__(iterations, epochs=epochs)

    def setRate(self):
        return self.min_lr * (self.lr_mult ** self.trn_iterations)

    def on_train_begin(self, logs={}):
        super().on_train_begin(logs=logs)
        try:  # multiple lr's
            K.get_variable_shape(self.model.optimizer.lr)[0]
            self.min_lr = np.full(K.get_variable_shape(self.model.optimizer.lr), self.min_lr)
        except IndexError:
            pass
        K.set_value(self.model.optimizer.lr, self.min_lr)
        self.best = 1e9
        self.model.save_weights('tmp.hd5')  # save weights

    def on_train_end(self, logs=None):
        self.model.load_weights('tmp.hd5')  # load_weights

    def on_batch_end(self, batch, logs=None):
        # check if we have made an x-fold jump in loss and training should stop
        try:
            loss = self.history['loss'][-1]
            if math.isnan(loss) or loss > self.best * self.jump:
                self.model.stop_training = True
            if loss < self.best:
                self.best = loss
        except KeyError:
            pass
        super().on_batch_end(batch, logs=logs)

class LR_Cycle(LR_Updater):
    '''This callback is utilized to implement cyclical learning rates
    it is based on this pytorch implementation https://github.com/fastai/fastai/blob/master/fastai
    and adopted from this keras implementation https://github.com/bckenstler/CLR
    it loosely implements methods described in the paper https://arxiv.org/pdf/1506.01186.pdf
    '''

    def __init__(self, iterations, cycle_len=1, cycle_mult=1, epochs=1):
        '''
        iterations = dataset size / batch size
        epochs #todo do i need this or can it accessed through self.model
        cycle_len = num of times learning rate anneals from its max to its min in an epoch
        cycle_mult = used to increase the cycle length cycle_mult times after every cycle
        for example: cycle_mult = 2 doubles the length of the cycle at the end of each cy$
        '''
        self.min_lr = 0
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult
        self.cycle_iterations = 0.
        super().__init__(iterations, epochs=epochs)

    def setRate(self):
        self.cycle_iterations += 1
        cos_out = np.cos(np.pi * (self.cycle_iterations) / self.epoch_iterations) + 1
        if self.cycle_iterations == self.epoch_iterations:
            self.cycle_iterations = 0.
            self.epoch_iterations *= self.cycle_mult
        return self.max_lr / 2 * cos_out

    def on_train_begin(self, logs={}):
        super().on_train_begin(logs={})  # changed to {} to fix plots after going from 1 to mult. lr
        self.cycle_iterations = 0.
        self.max_lr = K.get_value(self.model.optimizer.lr)

from keras import backend as K
from keras.callbacks import TensorBoard

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.get_value(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)