import pandas as pd
import torch
from ignite.engine import (Events, create_supervised_evaluator, create_supervised_trainer)
from ignite.metrics import Metric

from ..utils import (ChunkIter, data_gen_from_anno_gen, debug, gen_cwd_slash, gen_even_batches, preview_generator)


class RunningAverage:

    def __init__(self, src=None, alpha=0.98, get_value=lambda engine: engine.state.output):
        self._src = src
        self._alpha = alpha
        self._get_value = get_value
        self._value = None

    def update(self, value):
        if self._value is None:
            self._value = (1.0 - self._alpha) * value
        else:
            self._value = self._value * self._alpha + (1.0 - self._alpha) * value

    @torch.no_grad()
    def iteration_completed(self, engine, name):
        self.update(self._get_value(engine))
        engine.state.metrics[name] = self._value

    def attach(self, engine, name):
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed, name)


class MacroF1(Metric):

    def __init__(self, *args, epsilon=1e-7, threshold=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.threshold = threshold

    def reset(self):
        self.y_pred_list = []
        self.y_list = []

    def update(self, output):
        y_pred, y = output
        self.y_pred_list.append(y_pred)
        self.y_list.append(y)

    def compute(self):
        y_pred = torch.cat(self.y_pred_list)
        if self.threshold is not None:
            y_pred = torch.where(y_pred < self.threshold, torch.zeros_like(y_pred), torch.ones_like(y_pred))
        y = torch.cat(self.y_list)

        tp = torch.sum(y * y_pred, 0)
        # tn = torch.sum((1 - y) * (1 - y_pred), 0)
        fp = torch.sum((1 - y) * y_pred, 0)
        fn = torch.sum(y * (1 - y_pred), 0)

        p = tp / (tp + fp + self.epsilon)
        r = tp / (tp + fn + self.epsilon)

        f1s = 2 * p * r / (p + r + self.epsilon)

        macro_f1 = torch.mean(f1s)

        details = pd.DataFrame(
            {
                'true_positives': tp.cpu(),
                'false_positives': fp.cpu(),
                'false_negatives': fn.cpu(),
                'precision': p.cpu(),
                'recall': r.cpu(),
                'f1_score': f1s.cpu(),
            }
        )

        return {
            'score': macro_f1,
            'details': details,
        }


class CollectedPrediction(Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        self.y_pred_list = []
        self.y_list = []

    def update(self, output):
        y_pred, y = output
        self.y_pred_list.append(y_pred)

    def compute(self):
        return torch.cat(self.y_pred_list)
