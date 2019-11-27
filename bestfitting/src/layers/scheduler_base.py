class SchedulerBase(object):
    def __init__(self):
        self._is_load_best_weight = True
        self._is_load_best_optim = True
        self._is_freeze_bn=False
        self._is_adjust_lr = True
        self._lr = 0.01
        self._cur_optimizer = None

    def schedule(self, net, epoch, epochs, **kwargs):
        raise Exception('Did not implemented')

    def step(self, net, epoch, epochs):
        optimizer, lr = self.schedule(net, epoch, epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        lr_list = []
        for param_group in optimizer.param_groups:
            lr_list += [param_group['lr']]
        return lr_list

    def is_load_best_weight(self):
        return self._is_load_best_weight

    def is_load_best_optim(self):
        return self._is_load_best_optim

    def is_freeze_bn(self):
        return self._is_freeze_bn

    def reset(self):
        self._is_load_best_weight = True
        self._load_best_optim = True
        self._is_freeze_bn = False

    def is_adjust_lr(self):
        return self._is_adjust_lr
