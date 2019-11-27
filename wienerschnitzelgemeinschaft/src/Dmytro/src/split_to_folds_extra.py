import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from config import NB_FOLDS as nb_folds
import numpy as np
import config

training_samples = pd.read_csv('../input/data_extra/HPAv18RGBY_WithoutUncertain.csv')
training_samples['fold'] = -1

file_exists = []
file_names = []

for name in training_samples['Id']:
    img_id = name.split('_', 1)[1]
    exists = np.all([os.path.exists(f'../input/data_extra/img_512/{img_id}_{color}.png') for color in config.COLORS])
    file_exists.append(exists)
    if exists:
        file_names.append(img_id)

print(np.sum(np.array(file_exists)), np.sum(~np.array(file_exists)))

training_samples = training_samples[file_exists].reset_index(drop=True)
training_samples['Id'] = file_names

X = training_samples['Id'].tolist()
y = []

for name, lbl in zip(training_samples['Id'], training_samples['Target'].str.split(' ')):
        label = np.zeros(28)
        for key in lbl:
            label[int(key)] = 1
        y.append(label)

skf = MultilabelStratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print("train:", train_index, "test:", test_index)
    training_samples.loc[test_index, 'fold'] = fold

training_samples.to_csv(f'../input/folds_{nb_folds}_extra.csv', index=False)
