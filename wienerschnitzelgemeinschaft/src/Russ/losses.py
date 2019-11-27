import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
def smooth_l1_on_logits(logits, labels, reduction='none'):
    return F.smooth_l1_loss(torch.sigmoid(logits), labels, reduction=reduction)
  
class Smooth_L1_LossW(nn.Module):
    def __init__(self, pos_weight=1, reduce=True):
        super(Smooth_L1_LossW, self).__init__()
        self.pos_weight = pos_weight
        self.reduce = reduce

    def forward(self, logits, labels):
        l1 = smooth_l1_on_logits(logits, labels, reduction='none')
        w = 1. + labels * self.pos_weight
        l1 = l1 * w

        if self.reduce:
            return torch.mean(l1)
        else:
            return l1
        
def l1_on_logits(logits, labels, reduction='none'):
    return F.l1_loss(torch.sigmoid(logits), labels, reduction=reduction)
  
class L1_LossW(nn.Module):
    def __init__(self, pos_weight=1, reduce=True):
        super(L1_LossW, self).__init__()
        self.pos_weight = pos_weight
        self.reduce = reduce

    def forward(self, logits, labels):
        l1 = l1_on_logits(logits, labels, reduction='none')
        w = 1. + labels * self.pos_weight
        l1 = l1 * w

        if self.reduce:
            return torch.mean(l1)
        else:
            return l1


class BCELoss2d(nn.Module):
    """
    Binary Cross Entropy loss function
    """
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)
        return self.bce_loss(logits_flat, labels_flat)


class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        logits = logits.view(-1)
        gt = labels.view(-1)
        # http://geek.csdn.net/news/detail/126833
        loss = logits.clamp(min=0) - logits * gt + torch.log(1 + torch.exp(-logits.abs()))
        loss = loss * w
        loss = loss.sum() / w.sum()
        return loss


class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = torch.sigmoid(logits)
        num = labels.size(0)
        w = weights.view(num, -1)
        w2 = w * w
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * ((w2 * intersection).sum(1) + 1) / (
            (w2 * m1).sum(1) + (w2 * m2).sum(1) + 1)
        score = 1 - score.sum() / num
        return score


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        num = labels.size(0)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        # score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = (2.* intersection.sum(1) + 1.) / (m1.sum(1) + m2.sum(1) + 1.)
        score = 1. - score.sum() / num
        return score

class SoftDiceLoss0(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        num = labels.size(0)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        # score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = (2.* intersection.sum(1)) / (m1.sum(1) + m2.sum(1) + 1e-6)
        score = 1. - score.sum() / num
        return score

def dice_loss(input, target):
    smooth = 1.
    loss = 0.
    num_classes = 28
    for c in range(num_classes):
        iflat = input[:, c ].view(-1)
        tflat = target[:, c].view(-1)
        intersection = (iflat * tflat).sum()
        # w = class_weights[c]
        # loss += w*(1 - ((2. * intersection + smooth) /
        #         (iflat.sum() + tflat.sum() + smooth)))
        loss += (1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth)))
    return loss/num_classes


class DiceScore(nn.Module):
    def __init__(self, threshold=0.5):
        super(DiceScore, self).__init__()
        self.threshold = threshold

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        num = labels.size(0)
        predicts = (probs.view(num, -1) > self.threshold).float()
        labels = labels.view(num, -1)
        intersection = (predicts * labels)
        score = 2. * (intersection.sum(1)) / (predicts.sum(1) + labels.sum(1))
        return score.mean()


def dice_score_np(predicted_masks, labels):
    assert len(predicted_masks.shape) >= 3
    num = predicted_masks.shape[0]

    predicted_masks = predicted_masks.reshape(num, -1).astype(float)
    assert predicted_masks.min() == 0 and predicted_masks.max() == 1.0
    labels = labels.reshape(num, -1)
    assert labels.min() == 0 and labels.max() == 1.0
    intersection = (predicted_masks * labels)
    assert not np.any((predicted_masks.sum(1) + labels.sum(1)) == 0)
    score = 2. * (intersection.sum(1)) / (predicted_masks.sum(1) + labels.sum(1))
    assert len(score.shape) == 1
    return score.mean()


class CombinedLoss(nn.Module):
    def __init__(self, is_weight=True, is_log_dice=False):
        super(CombinedLoss, self).__init__()
        self.is_weight = is_weight
        self.is_log_dice = is_log_dice
        if self.is_weight:
            self.weighted_bce = WeightedBCELoss2d()
            self.soft_weighted_dice = WeightedSoftDiceLoss()
        else:
            self.bce = BCELoss2d()
            self.soft_dice = SoftDiceLoss()

    def forward(self, logits, labels):
        size = logits.size()
        # assert size[1] == 1, size
        # logits = logits.view(size[0], size[2], size[3])
        # labels = labels.view(size[0], size[2], size[3])
        if self.is_weight:
            batch_size, H, W = labels.size()
            if H == 128:
                kernel_size = 11
            elif H == 256:
                kernel_size = 21
            elif H == 512:
                kernel_size = 21
            elif H == 1024:
                kernel_size = 41
            elif H == 1280:
                kernel_size = 51
            else:
                raise ValueError('Unknown height')

            a = F.avg_pool2d(labels, kernel_size=kernel_size, padding=kernel_size // 2,
                             stride=1)
            ind = a.ge(0.01) * a.le(0.99)
            ind = ind.float()
            # weights = Variable(torch.tensor.torch.ones(a.size())).cuda()
            weights = Variable(torch.ones(a.size())).cuda()

            w0 = weights.sum()
            weights += ind * 2
            w1 = weights.sum()
            weights = weights / w1 * w0

            bce_loss = self.weighted_bce(logits, labels, weights)
            dice_loss = self.soft_weighted_dice(logits, labels, weights)
        else:
            bce_loss = self.bce(logits, labels)
            dice_loss = self.soft_dice(logits, labels)

        if self.is_log_dice:
            l = bce_loss - (1 - dice_loss).log()
        else:
            l = bce_loss + dice_loss
        return l, bce_loss, dice_loss


def combined_loss(logits, labels, is_weight=True, is_log_dice=False):
    size = logits.size()
    assert size[1] == 1, size
    logits = logits.view(size[0], size[2], size[3])
    labels = labels.view(size[0], size[2], size[3])
    if is_weight:
        batch_size, H, W = labels.size()
        if H == 128:
            kernel_size = 11
        elif H == 256:
            kernel_size = 21
        elif H == 512:
            kernel_size = 21
        elif H == 1024:
            kernel_size = 41
        elif H == 1280:
            kernel_size = 51
        else:
            raise ValueError('Unknown height')

        a = F.avg_pool2d(labels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        ind = a.ge(0.01) * a.le(0.99)
        ind = ind.float()
        weights = Variable(torch.tensor.torch.ones(a.size())).cuda()

        w0 = weights.sum()
        weights += ind * 2
        w1 = weights.sum()
        weights = weights / w1 * w0

        bce_loss = WeightedBCELoss2d().cuda()(logits, labels, weights)
        dice_loss = WeightedSoftDiceLoss().cuda()(logits, labels, weights)
    else:
        bce_loss = BCELoss2d().cuda()(logits, labels)
        dice_loss = SoftDiceLoss().cuda()(logits, labels)

    if is_log_dice:
        l = bce_loss - (1 - dice_loss).log()
    else:
        l = bce_loss + dice_loss
    return l, bce_loss, dice_loss

class ThreeWayLoss(nn.Module):
    def __init__(self):
        super(ThreeWayLoss, self).__init__()
        
        self.focal = FocalLoss()
        self.bce = BCELoss2d()
        self.dice = SoftDiceLoss()

    def forward(self, logits, labels):
        
        focal_loss = self.focal(logits, labels)
        bce_loss = self.bce(logits, labels)
        dice_loss = self.dice(logits, labels)

        loss = focal_loss + bce_loss + dice_loss
        return loss


