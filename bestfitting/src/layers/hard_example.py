import torch

def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels

def get_hard_samples(logits,labels,neg_more=2,neg_least_ratio=0.5,neg_max_ratio=0.7):
    logits = logits.view(-1)
    labels = labels.view(-1)

    pos_idcs = labels > 0
    pos_output = logits[pos_idcs]
    pos_labels = labels[pos_idcs]

    neg_idcs = labels <= 0
    neg_output = logits[neg_idcs]
    neg_labels = labels[neg_idcs]

    neg_at_least=max(neg_more,int(neg_least_ratio * neg_output.size(0)))
    hard_num = min(neg_output.size(0),pos_output.size(0) + neg_at_least, int(neg_max_ratio * neg_output.size(0)) + neg_more)
    if hard_num > 0:
        neg_output, neg_labels = hard_mining(neg_output, neg_labels, hard_num)

    logits=torch.cat([pos_output,neg_output])
    labels = torch.cat([pos_labels, neg_labels])


    return logits,labels