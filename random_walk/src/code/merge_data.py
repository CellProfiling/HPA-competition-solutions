import pandas as pd
import os

# original_csv = pd.read_csv('../input/train.csv')
# external_csv = pd.read_csv('../input/HPAv18RGBY_WithoutUncertain_wodpl.csv')

# train_csv = pd.concat([original_csv,external_csv], sort=False)
# train_csv.to_csv('../input/train_merge.csv', header=True, index=False)
img_list = os.listdir('../input/train/')
path = '../input/train/'
num = 0
for img_name in img_list:
    tokens = img_name.split('.')
    if tokens[-1] == 'jpg':
        print(tokens)
        num += 1
        # print(num)
        os.rename(path+img_name, path+tokens[0] + '.png')
print(num)