from utils.include import *
from utils.loss import *
from utils.metric import accuracy
from utils.batchnorm import SynchronizedBatchNorm1d

class Resnet101(nn.Module):
    """docstring for Resnet101"""
    def __init__(self, num_classes=28):
        super(Resnet101, self).__init__()
        self.num_classes = num_classes
        self.resnet = torchvision.models.resnet101(pretrained=True)
        conv1_weight = self.resnet.conv1.weight.data
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        # self.resnet.fc = nn.Sequential(
        #     nn.Linear(self.resnet.fc.in_features, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(2048, num_classes),
        #     ) # not work
        self.resnet.conv1.weight = torch.nn.Parameter(torch.cat((conv1_weight,conv1_weight[:,:1,:,:]),dim=1))

    def forward(self, x):
        x = self.resnet(x)
        return x

    def criterion(self, logit, truth ):
        # loss = FocalLoss2d()(logit, truth, type='sigmoid')
        loss = F1_loss()(logit, truth)
        return loss

    def metric(self, logit, truth, threshold=0.5):
        prob = F.sigmoid(logit)
        acc = accuracy(prob, truth, threshold=threshold, is_average=True)
        return acc

class Resnet50(nn.Module):
    """docstring for Resnet50"""
    def __init__(self, num_classes=28):
        super(Resnet50, self).__init__()
        self.num_classes = num_classes
        self.resnet = torchvision.models.resnet50(pretrained=True)
        conv1_weight = self.resnet.conv1.weight.data
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        # self.resnet.fc = nn.Sequential(
        #     nn.Linear(self.resnet.fc.in_features, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(2048, num_classes),
        #     ) # not work
        self.resnet.conv1.weight = torch.nn.Parameter(torch.cat((conv1_weight,conv1_weight[:,:1,:,:]),dim=1))
        # for m in self.resnet.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()
        #         # shutdown update in frozen mode
        #         m.weight.requires_grad = False
        #         m.bias.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        return x

    def criterion(self, logit, truth ):
        # loss = FocalLoss2d()(logit, truth, type='sigmoid')
        # weights = np.ones(28, dtype=np.float32) * 10
        loss = F1_loss()(logit, truth)
        # bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(weights).cuda())(logit, truth)
        # loss = f1_loss + bce_loss
        return loss

    def metric(self, logit, truth, threshold=0.5):
        prob = F.sigmoid(logit)
        acc = accuracy(prob, truth, threshold=threshold, is_average=True)
        return acc


class Resnet34(nn.Module):
    """docstring for Resnet34"""
    def __init__(self, num_classes=28):
        super(Resnet34, self).__init__()
        self.num_classes = num_classes
        self.resnet = torchvision.models.resnet34(pretrained=True)
        conv1_weight = self.resnet.conv1.weight.data
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        # self.resnet.fc = nn.Sequential(
        #     nn.Linear(self.resnet.fc.in_features, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(2048, num_classes),
        #     ) # not work
        self.resnet.conv1.weight = torch.nn.Parameter(torch.cat((conv1_weight,conv1_weight[:,:1,:,:]),dim=1))

    def forward(self, x):
        x = self.resnet(x)
        return x

    def criterion(self, logit, truth ):
        # loss = FocalLoss2d()(logit, truth, type='sigmoid')
        loss = F1_loss()(logit, truth)
        return loss

    def metric(self, logit, truth, threshold=0.5):
        prob = F.sigmoid(logit)
        acc = accuracy(prob, truth, threshold=threshold, is_average=True)
        return acc

class Resnet18(nn.Module):
    """docstring for Resnet18"""
    def __init__(self, num_classes=28):
        super(Resnet18, self).__init__()
        self.num_classes = num_classes
        self.resnet = torchvision.models.resnet18(pretrained=True)
        conv1_weight = self.resnet.conv1.weight.data
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        # self.resnet.fc = nn.Sequential(
        #     nn.Linear(self.resnet.fc.in_features, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(2048, num_classes),
        #     ) # not work
        self.resnet.conv1.weight = torch.nn.Parameter(torch.cat((conv1_weight,conv1_weight[:,:1,:,:]),dim=1))

    def forward(self, x):
        x = self.resnet(x)
        return x

    def criterion(self, logit, truth ):
        # loss = FocalLoss2d()(logit, truth, type='sigmoid')
        loss = F1_loss()(logit, truth)
        return loss

    def metric(self, logit, truth, threshold=0.5):
        prob = F.sigmoid(logit)
        acc = accuracy(prob, truth, threshold=threshold, is_average=True)
        return acc

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

class Resnet50Base(nn.Module):
    """docstring for Resnet50Base"""
    def __init__(self, num_classes=28):
        super(Resnet50Base, self).__init__()
        self.num_classes = num_classes
        self.resnet = torchvision.models.resnet50(pretrained=True)
        conv1_weight = self.resnet.conv1.weight.data
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.resnet.avgpool = AdaptiveConcatPool2d(1)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            SynchronizedBatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
            )
        self.resnet.conv1.weight = torch.nn.Parameter(torch.cat((conv1_weight,conv1_weight[:,:1,:,:]),dim=1))

    def forward(self, x):
        x = self.resnet(x)
        return x

    def criterion(self, logit, truth ):
        loss = FocalLoss2d()(logit, truth, type='sigmoid')
        return loss

    def metric(self, logit, truth, threshold=0.5):
        prob = F.sigmoid(logit)
        acc = accuracy(prob, truth, threshold=threshold, is_average=True)
        return acc


def run_check_net():
    batch_size = 8
    C, H, W = 1, 512, 512

    input = np.random.uniform(0, 1, (batch_size, C, H, W)).astype(np.float32)
    truth = np.random.choice(2, (batch_size, C, H, W)).astype(np.float32)

    # ------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).float().cuda()

    # ---
    net = Resnet50().cuda()
    # net.set_mode('train')
    net.train()
    # print(net)
    # exit(0)

    # net.load_pretrain('/root/share/project/kaggle/tgs/data/model/resnet50-19c8e357.pth')

    logit = net(input)
    loss = net.criterion(logit, truth)
    dice = net.metric(logit, truth)

    print('loss : %0.8f' % loss.item())
    print('dice : %0.8f' % dice.item())
    print('')

    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.001, momentum=0.9, weight_decay=0.0001)

    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    i = 0
    optimizer.zero_grad()
    while i <= 500:

        logit = net(input)
        loss = net.criterion(logit, truth)
        dice = net.metric(logit, truth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 20 == 0:
            print('[%05d] loss, dice  :  %0.5f,%0.5f' % (i, loss.item(), dice.item()))
        i = i + 1


def plot_net():
    batch_size = 8
    C, H, W = 1, 256, 256
    dummy_input = torch.rand(batch_size, C, H, W)
    net = SaltNet()
    net.set_mode('train')

    with SummaryWriter(comment='LeNet') as w:
        w.add_graph(net, (dummy_input,))


########################################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()
    # plot_net()
    print('sucessful!')
