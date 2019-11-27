import pandas as pd
import numpy as np

def sub2labels(sub):
    x = sub['Predicted'].fillna('-1').astype(str).apply(lambda x: x.split())
    x = [[int(label) for label in labels] for labels in x]
    labels1 = np.zeros((len(x),28))
    for i in range(len(x)):
        labels1[i][x[i]] = 1
    return labels1

def sub_df_corr(sub1_df,sub2_df):

    x1 = sub2labels(sub1_df)
    x2 = sub2labels(sub2_df)

    print(np.corrcoef(x1.flatten(),x2.flatten()))


'17/submission_base_best_val_0.6405-0.6583.csv'

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix


# computute confusion matrices between two submission files

def f1_confusion(sub1_df, sub2_df, num_classes=28):
    c0 = sub1_df
    c1 = sub2_df
    assert c0.shape == c1.shape
    s0 = [s if isinstance(s, str) else '' for s in c0.Predicted]
    s1 = [s if isinstance(s, str) else '' for s in c1.Predicted]
    p0 = [s.split() for s in s0]
    p1 = [s.split() for s in s1]
    y0 = np.zeros((c0.shape[0], num_classes)).astype(int)
    y1 = np.zeros((c0.shape[0], num_classes)).astype(int)
    # print(p0[:5])
    for i in range(c0.shape[0]):
        for j in p0[i]: y0[i, int(j)] = 1
        for j in p1[i]: y1[i, int(j)] = 1
    # print(y0[:5])
    y0avg = np.average(y0, axis=0)
    y1avg = np.average(y1, axis=0)
    cm = [confusion_matrix(y0[:, i], y1[:, i]) for i in range(y0.shape[1])]
    fm = [f1_score(y0[:, i], y1[:, i]) for i in range(y0.shape[1])]
    for i in range(y0.shape[1]):
        print(LABEL_MAP[i])
        print(cm[i], ' %4.2f' % fm[i], ' %6.4f' % y0avg[i], ' %6.4f' % y1avg[i],
              ' %6.4f' % (y0avg[i] - y1avg[i]))
        print()

    print('f1 macro')
    print(np.mean(fm))

    return f1_score(y0, y1, average='macro')

# compute f1 score between two submission files
from sklearn.metrics import f1_score
def f1_sub(sub1_df, sub2_df, num_classes=28):
    c0 = sub1_df
    c1 = sub2_df
    assert c0.shape == c1.shape
    s0 = [s if isinstance(s, str) else '' for s in c0.Predicted]
    s1 = [s if isinstance(s, str) else '' for s in c1.Predicted]
    p0 = [s.split() for s in s0]
    p1 = [s.split() for s in s1]
    y0 = np.zeros((c0.shape[0], num_classes)).astype(int)
    y1 = np.zeros((c0.shape[0], num_classes)).astype(int)
    # print(p0[:5])
    for i in range(c0.shape[0]):
        for j in p0[i]: y0[i, int(j)] = 1
        for j in p1[i]: y1[i, int(j)] = 1
    # print(y0[:5])
    return f1_score(y0, y1, average='macro')
