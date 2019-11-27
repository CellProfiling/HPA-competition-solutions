import pandas as pd

LABELS_ext = 'input/HPAv18/HPAv18RBGY_wodpl.csv'
LABELS = 'input/train.csv'

df1 = pd.read_csv(LABELS)
df2 = pd.read_csv(LABELS_ext)

df3  = pd.concat([df1,df2], axis=0)
df3.to_csv('input/train_org_ext.csv', index=False)

print('success')