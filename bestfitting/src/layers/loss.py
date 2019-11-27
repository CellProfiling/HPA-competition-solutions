import math

from torch import nn
from config.config import *
from layers.hard_example import *
from layers.lovasz_losses import *

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target, epoch=0):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()

class HardLogLoss(nn.Module):
    def __init__(self):
        super(HardLogLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.__classes_num = NUM_CLASSES

    def forward(self, logits, labels,epoch=0):
        labels = labels.float()
        loss=0
        for i in range(NUM_CLASSES):
            logit_ac=logits[:,i]
            label_ac=labels[:,i]
            logit_ac, label_ac=get_hard_samples(logit_ac,label_ac)
            loss+=self.bce_loss(logit_ac,label_ac)
        loss = loss/NUM_CLASSES
        return loss

# https://github.com/bermanmaxim/LovaszSoftmax/tree/master/pytorch
def lovasz_hinge(logits, labels, ignore=None, per_class=True):
    """
    Binary Lovasz hinge loss
      logits: [B, C] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, C] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_class:
        loss = 0
        for i in range(NUM_CLASSES):
            logit_ac = logits[:, i]
            label_ac = labels[:, i]
            loss += lovasz_hinge_flat(logit_ac, label_ac)
        loss = loss / NUM_CLASSES
    else:
        logits = logits.view(-1)
        labels = labels.view(-1)
        loss = lovasz_hinge_flat(logits, labels)
    return loss

# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053
class SymmetricLovaszLoss(nn.Module):
    def __init__(self):
        super(SymmetricLovaszLoss, self).__init__()
        self.__classes_num = NUM_CLASSES

    def forward(self, logits, labels,epoch=0):
        labels = labels.float()
        loss=((lovasz_hinge(logits, labels)) + (lovasz_hinge(-logits, 1 - labels))) / 2
        return loss

class FocalSymmetricLovaszHardLogLoss(nn.Module):
    def __init__(self):
        super(FocalSymmetricLovaszHardLogLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.slov_loss = SymmetricLovaszLoss()
        self.log_loss = HardLogLoss()
    def forward(self, logit, labels,epoch=0):
        labels = labels.float()
        focal_loss = self.focal_loss.forward(logit, labels, epoch)
        slov_loss = self.slov_loss.forward(logit, labels, epoch)
        log_loss = self.log_loss.forward(logit, labels, epoch)
        loss = focal_loss*0.5 + slov_loss*0.5 +log_loss * 0.5
        return loss

# https://github.com/ronghuaiyang/arcface-pytorch
class ArcFaceLoss(nn.modules.Module):
    def __init__(self,s=30.0,m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels, epoch=0):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cosine, labels)
        gamma=1
        loss=(loss1+gamma*loss2)/(1+gamma)
        return loss
