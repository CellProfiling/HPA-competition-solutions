lb_dist = {

0 : 0.36239782,
1 : 0.043841336,
2 : 0.075268817,
3 : 0.059322034,
4 : 0.075268817,
5 : 0.075268817,
6 : 0.043841336,
7 : 0.075268817,
8 : 0,
9 : 0,
10 : 0,
11 : 0.043841336,
12 : 0.043841336,
13 : 0.014198783,
14 : 0.043841336,
15 : 0,
16 : 0.028806584,
17 : 0.014198783,
18 : 0.028806584,
19 : 0.059322034,
20 : 0,
21 : 0.126126126,
22 : 0.028806584,
23 : 0.075268817,
24 : 0,
25 : 0.222493888,
26 : 0.028806584,
27 : 0
}




import numpy as np

lb_dist2 = np.array([lb_dist[id] for id in list(range(28))])
lb_dist2[lb_dist2 == 0] = 0.001

import pandas as pd
data = pd.read_csv('Christof/assets/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append(np.array([int(label) for label in labels]))

counts = np.zeros(28)
for item in train_dataset_info:
    for l in item:
        counts[l] = counts[l]+1

counts /= len(train_dataset_info)

weights = lb_dist2 / counts

label_dist = pd.DataFrame(lb_dist2)
label_dist['train_dist'] = counts
label_dist.columns = ['lb','train']
label_dist.to_csv('label_dist.csv',index=False)