import numpy as np
from sklearn.metrics.ranking import roc_auc_score
import torch

# compute auc for each class
def computeAUROC(dataGT, dataPRED, classCount):

    outAUROC = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

    return outAUROC

# search the threshold which was used to obtain the maximum f1 score for each class
def search_f1(output, target):
    max_result_f1_list = []
    max_threshold_list = []
    eps=1e-20
    target = target.type(torch.cuda.ByteTensor)

    for i in range(output.shape[1]):

        output_class = output[:, i]
        target_class = target[:, i]
        max_result_f1 = 0
        max_threshold = 0

        for threshold in [x * 0.01 for x in range(0, 100)]:

            prob = output_class > threshold
            label = target_class
            TP = (prob & label).sum().float()
            TN = ((~prob) & (~label)).sum().float()
            FP = (prob & (~label)).sum().float()
            FN = ((~prob) & label).sum().float()

            precision = TP / (TP + FP + eps)
            recall = TP / (TP + FN + eps)
            result_f1 = 2 * precision  * recall / (precision + recall + eps)

            if result_f1.item() > max_result_f1:
                max_result_f1 = result_f1.item()
                max_threshold = threshold

        max_result_f1_list.append(round(max_result_f1,3))
        max_threshold_list.append(max_threshold)

    return max_threshold_list, max_result_f1_list