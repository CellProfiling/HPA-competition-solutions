import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class BinaryEntropyLoss_weight(nn.Module):
    def __init__(self, weight=None, size_average=True, is_weight=True):
        super(BinaryEntropyLoss_weight, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.is_weight = is_weight
        # self.class_num = np.array([[12885, 1254, 3621, 1561, 1858, 2513, 1008, 2822, 53, 45, 28, 1093, 688, 537, 1066, 21, 530, 210, 902, 1482, 172, 3777, 802, 2965, 322, 8228, 328, 11]])
        self.class_num = np.array([[12885, 1254, 3621, 1561/2, 1858, 2513/2, 1008/2, 2822, 53, 45, 28, 1093, 688, 537, 1066, 21, 530, 210, 902, 1482/2, 172, 3777/2, 802, 2965, 322, 8228/2, 328, 11]])
        
        # self.class_num = np.array([[46914, 2495, 8877, 2979, 4330, 4665, 2901, 7222, 159, 116, 105, 1704, 1625, 1041, 2279, 37, 877, 267, 1436, 2911, 319, 10691, 1789, 8589, 472, 31726, 593, 53]])
        # self.class_num = np.array([[46914, 2495, 8877, 2979, 4330, 4665, 2901, 7222, 159, 116, 105, 1704, 1625, 1041, 2279, 37, 877, 267, 1436, 2911, 319, 10691, 1789, 8589, 472, 31726, 593, 53]])
        self.class_num = np.power((1-self.class_num/30000), 2)
        # print(target.shape)

    def forward(self, input, target):

        # if self.is_weight:
        #     # class_num = np.array([[12885, 1254, 3621, 1561, 1858, 2513, 1008, 2822, 53, 45, 28, 1093, 688, 537, 1066, 21, 530, 210, 902, 1482, 172, 3777, 802, 2965, 322, 8228, 328, 11]])
        #     # class_num = np.array([[23374, 2068, 5527, 2239, 3248, 3879, 2018, 5536, 87, 63, 28, 1717, 1468, 983, 1982, 21, 932, 292, 1312, 2595, 288, 7973, 1430, 5588, 382, 20099, 512, 109]])
        #     # class_num = np.array([[24910,2096,5780,2297,3350,4027,2178,5795,111,73,56,1727,1564,1033,2026,27,954,344,1381,2711,298,8718,1628,5940,382,21918,530,113]])
        #     # class_num = np.array([[40581, 3045, 10697, 3303, 5081, 5888, 3849, 9079, 211, 199, 184, 2105, 2128, 1422, 2622, 63, 1254, 434, 1826, 3600, 442, 13508, 2663, 10142, 408, 36664, 692, 137]])
        #     # class_num = np.array([[39130, 2742, 9311, 3232, 4775, 4962, 3299, 7611, 189, 139, 122, 1850, 1730, 1137, 2474, 41, 878, 288, 1495, 3123, 352, 11332, 1964, 9098, 510, 32952, 646, 59]])
        #     class_num = np.array([[49315, 2776, 9455, 3254, 4821, 5020, 3333, 7709, 189, 139, 122, 1858, 1762, 1157, 2524, 41, 1033, 290, 1531, 3137, 356, 11444, 1976, 9212, 510, 33555, 646, 59]])
        #     self.weight = class_num
        #     self.weight = np.power((1-self.weight/100000), 2)
        #     # print(target.shape)
        #     self.weight = torch.cuda.FloatTensor(self.weight.repeat(target.shape[0], axis=0))
        self.weight = torch.cuda.FloatTensor(self.class_num.repeat(target.shape[0], axis=0))

        loss = F.binary_cross_entropy(input, target, self.weight, self.size_average)

        return loss
