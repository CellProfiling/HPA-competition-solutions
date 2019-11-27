from tensorpack import HyperParamSetter
import operator
import numpy as np
import tensorflow as tf
from tensorpack.utils import logger
from collections import deque

class ReduceLearningRateOnPlateau(HyperParamSetter):
    """
    Cyclic Learning Rate: https://arxiv.org/pdf/1506.01186.pdf
    mode: triangular2 or exp_range. detail in paper.
    """

    def __init__(self, param, base_lr=0.01, factor=0.1, patience=5, min_lr=1e-5, window_size=200):
        self.patience = patience
        self.base_lr = base_lr
        self.factor = factor
        self.min_lr = min_lr
        self.best = 1000000
        self.wait = 0
        self._window = int(window_size)
        self._queue = deque(maxlen=window_size)
        super(ReduceLearningRateOnPlateau, self).__init__(param)
    
    #def _setup_graph(self):
    #    tensors = tf.get_default_graph().get_tensor_by_name("tower0/cls_loss/label_loss:0")
    #    self._fetch = tf.train.SessionRunArgs(tensors)

    def _get_value_to_set(self):
        #label_loss = tf.get_default_graph().get_tensor_by_name("tower0/cls_loss/label_loss:0")
        #label_loss = label_loss.eval()
        if len(self._queue) > 0:
            moving_mean = np.asarray(self._queue).mean(axis=0)
        else:
            return self.base_lr
        if moving_mean < self.best:
            self.best = moving_mean
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.wait = 0
                self.base_lr = max(self.base_lr * self.factor, self.min_lr)
                logger.warn("ReduceLROnPlateau reducing learning rate to {}".format(self.base_lr))
        return self.base_lr

    def _trigger_epoch(self):
        self.trigger()

    def _before_run(self, _):
        return tf.train.SessionRunArgs(tf.get_default_graph().get_tensor_by_name("tower0/cls_loss/label_loss:0"))

    def _after_run(self, _, rv):
        results = rv.results
        self._queue.append(results)

    #def _trigger_step(self):
    #    label_loss = tf.get_default_graph().get_tensor_by_name("tower0/cls_loss/label_loss:0")
    #    self._queue.append(label_loss.eval())


class CyclicLearningRateSetter(HyperParamSetter):
    """
    Cyclic Learning Rate: https://arxiv.org/pdf/1506.01186.pdf
    mode: triangular2 or exp_range. detail in paper.
    """

    def __init__(self, param, base_lr=0.001, max_lr=0.006, step_size=2000., mode="triangular2", step_based=True):
        self._step = step_based
        self.mode = mode
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = float(step_size)
        self.scale_fn = lambda x: 1 / (2. ** (x - 1)) if mode == 'triangular2' else lambda x: gamma**(x)
        super(CyclicLearningRateSetter, self).__init__(param)

    def clr(self):
        cycle_num = np.floor(1 + float(self.global_step) / (2 * self.step_size))
        step_ratio = np.abs(float(self.global_step) / self.step_size - 2 * cycle_num + 1)
        if self.mode == 'triangular2':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-step_ratio)) * self.scale_fn(cycle_num)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-step_ratio)) * self.scale_fn(self.global_step)

    def _get_value_to_set(self):
        v = self.clr()
        return v

    def _trigger_epoch(self):
        if not self._step:
            self.trigger()

    def _trigger_step(self):
        if self._step:
            self.trigger()
