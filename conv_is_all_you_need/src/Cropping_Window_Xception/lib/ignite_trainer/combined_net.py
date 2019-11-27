import torch
from torch import nn
from .pytorch_mobilenet_v2_xzhu import MobileNetV2


class CombinedNet(nn.Module):

    def __init__(self, get_base_net, state_dicts):
        super().__init__()
        nets = []
        for state_dict in state_dicts:
            net = get_base_net()
            net.load_state_dict(state_dict)
            nets.append(net)
        self.nets = nn.ModuleList(nets)

    def forward(self, input):
        return torch.cat([net(input) for net in self.nets], dim=1)
