from utils.include import *

def accuracy(prob, truth, threshold=0.5,  is_average=True):
    batch_size = prob.size(0)
    p = prob.detach().view(batch_size,-1)
    t = truth.detach().view(batch_size,-1)

    p = p>threshold
    t = t>0.5
    correct = ( p == t).float()
    accuracy = correct.sum(1)/p.size(1)

    if is_average:
        accuracy = accuracy.sum()/batch_size
        return accuracy
    else:
        return accuracy

# def do_kaggle_metric(predicts, truths):
#     # thresholds = np.linspace(0, 1, 1500)
#     thresholds = np.arange(0, 1, 0.001)
#     score = 0.0
#     best_threshold=0.0
#     best_val = 0.0
#     for threshold in thresholds:
#         score = f1_score(truths > 0.5, predicts > threshold, average='macro')
#         if score > best_val:
#             best_threshold = threshold
#             best_val = score
#         # print("Threshold %0.4f, F1: %0.4f" % (threshold,score))

#     # print("BEST: %0.5f, F1: %0.5f" % (best_threshold,best_val))
#     return best_val, best_threshold

def do_kaggle_metric(predicts, truths):
    thresholds = np.arange(0, 1, 0.001)
    f1_score_matrix = np.zeros((thresholds.shape[0], 28))
    for j, threshold in enumerate(thresholds):
        for i in range(28):
            pred = np.array(predicts[:, i]>threshold, dtype=np.int8)
            f1_score_matrix[j,i] = f1_score(truths[:, i], pred, average='binary')
    best_f1_score = np.mean(np.max(f1_score_matrix, axis=0))
    best_threshold = np.empty(28)
    for i in range(28):
        best_threshold[i] = thresholds[np.where(f1_score_matrix[:,i] == np.max(f1_score_matrix[:,i]))[0][0]]
    return best_f1_score, best_threshold

