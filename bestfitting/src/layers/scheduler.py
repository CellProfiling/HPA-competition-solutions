import torch.optim as optim
from layers.scheduler_base import SchedulerBase

class Adam45(SchedulerBase):
    def __init__(self, params_list=None):
        super(Adam45, self).__init__()
        self._lr = 3e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 30e-5
        if epoch > 25:
            lr = 15e-5
        if epoch > 30:
            lr = 7.5e-5
        if epoch > 35:
            lr = 3e-5
        if epoch > 40:
            lr = 1e-5
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0005
        return self._cur_optimizer, self._lr


class Adam55(SchedulerBase):
    def __init__(self, params_list=None):
        super(Adam55, self).__init__()
        self._lr = 3e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self,net, epoch, epochs, **kwargs):
        lr = 30e-5
        if epoch > 25:
            lr = 15e-5
        if epoch > 35:
            lr = 7.5e-5
        if epoch > 45:
            lr = 3e-5
        if epoch > 50:
            lr = 1e-5
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.0005
        return self._cur_optimizer, self._lr

class FaceAdam(SchedulerBase):
    def __init__(self,params_list=None):
        super(FaceAdam, self).__init__()
        self._lr = 2e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 1e-4
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.0005
        return self._cur_optimizer, self._lr
