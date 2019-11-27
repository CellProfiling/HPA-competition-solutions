from utils.include import *
from utils.loss import *
import collections
# from .networks.sononet_grid_attention import sononet_grid_attention as model
from .networks.resnet_grid_attention import resnet_grid_attention as model
class AggregatedClassifier(nn.Module):
    def __init__(self, num_classes=28):
        super(AggregatedClassifier, self).__init__()
        weight = [1, 1, 1]  # copy
        weight_t = torch.from_numpy(np.array(weight, dtype=np.float32))
        self.weight = weight
        self.aggregation = "mean"
        self.aggregation_param = None
        self.aggregation_weight = weight_t.view(-1,1,1).cuda()
        self.net =  model(n_classes=num_classes,
                      is_batchnorm=True,
                      in_channels=4,
                      feature_scale=8,
                      nonlocal_mode="concatenation_mean_flow",
                      aggregation_mode="mean")
        self.mode = 'train'

    def compute_loss(self, logits, truth):
        """Compute loss function. Iterate over multiple output"""
        weights = self.weight
        if not isinstance(logits, collections.Sequence):
            logits = [logits]
            weights = [1]

        loss = 0
        for lmda, prediction in zip(weights, logits):
            if lmda == 0:
                continue
            loss += lmda * self.criterion(prediction, truth)

        self.loss = loss
        return loss

    def aggregate_output(self):
        """Given a list of predictions from net, make a decision based on aggreagation rule"""
        # if isinstance(self.predictions, collections.Sequence):
        #     logits = []
        #     for pred in self.predictions:
        #         logit = self.net.apply_argmax_softmax(pred).unsqueeze(0)
        #         logits.append(logit)

        #     logits = torch.cat(logits, 0)
        #     if self.aggregation == 'max':
        #         self.pred = logits.data.max(0)[0].max(1)
        #     elif self.aggregation == 'mean':
        #         self.pred = logits.data.mean(0).max(1)
        #     elif self.aggregation == 'weighted_mean':
        #         self.pred = (self.aggregation_weight.expand_as(logits) * logits).data.mean(0).max(1)
        #     elif self.aggregation == 'idx':
        #         self.pred = logits[self.aggregation_param].data.max(1)
        # else:
        #     # Apply a softmax and return a segmentation map
        #     self.logits = self.net.apply_argmax_softmax(self.predictions)
        #     self.pred = self.logits.data.max(1)

        if isinstance(self.predictions, collections.Sequence):
            logits = []
            for pred in self.predictions:
                logit = pred.unsqueeze(0)
                logits.append(logit)

            logits = torch.cat(logits, 0)
            if self.aggregation == 'max':
                self.pred = logits.max(0)
            elif self.aggregation == 'mean':
                self.pred = logits.mean(0)
            elif self.aggregation == 'weighted_mean':
                self.pred = (self.aggregation_weight.expand_as(logits) * logits).mean(0)
        else:
            # Apply a softmax and return a segmentation map
            self.pred = self.predictions
        return self.pred


    def forward(self, x):
        if self.mode in ['train']:
            self.predictions = self.net(x)
        elif self.mode in ['eval', 'valid', 'test']:
            self.predictions = self.net(x)
            self.predictions = self.aggregate_output()
        return self.predictions

    def criterion(self, logit, truth):
        # weights = np.ones(28, dtype=np.float32) * 10
        f1_loss = F1_loss()(logit, truth)
        focal_loss = FocalLoss()(logit, truth)
        # bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(weights).cuda())(logit, truth)
        # loss = nn.BCEWithLogitsLoss()(logit, truth)
        loss = f1_loss + focal_loss
        return loss

    # def backward(self):
    #     self.compute_loss()
    #     self.loss.backward()

    # def validate(self, x):
        # self.net.eval()
        # self.forward(split='test')
        # self.compute_loss()

    # def reset_results(self):
    #     self.losses = []
    #     self.pr_lbls = []
    #     self.pr_probs = []
    #     self.gt_lbls = []

    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError
