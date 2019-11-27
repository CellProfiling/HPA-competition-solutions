import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from config import NB_FOLDS as nb_folds
import numpy as np
from sklearn.cluster import KMeans
import config
import utils


def split_to_folds_ml():
    training_samples = pd.read_csv('../input/train.csv')
    training_samples['fold'] = -1
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

    training_samples.to_csv(f'../input/folds_{nb_folds}.csv', index=False)


def split_to_folds_emb():
    training_samples = pd.read_csv('../input/emb_train.csv')
    training_samples['fold'] = -1

    all_labels = []
    for _, row in training_samples.iterrows():
        lbl = [int(l) for l in row['Target'].split(' ')]
        labels = np.zeros(config.NB_CATEGORIES, dtype=np.float32)
        labels[lbl] = 1.0
        all_labels.append(labels)
    all_labels = np.array(all_labels)

    label_frequency = np.mean(all_labels, axis=0)
    label_order = np.argsort(label_frequency)[::-1]  # start from more common labels, so less common overrides it
    for label in label_order:
        group_df = training_samples[all_labels[:, label] > 0]
        ids = group_df.Id.values
        emb = group_df.as_matrix(columns=[f'emb_{i}' for i in range(config.NB_EMBEDDINGS)])

        n_samples = len(group_df)
        clusters = KMeans(n_clusters=min(4, n_samples), random_state=0, max_iter=800, n_jobs=1, verbose=0).fit_predict(emb)

        print(label, n_samples, [np.sum(clusters == i) for i in range(4)])
        print(np.std(emb, axis=0))

        for fold in range(config.NB_FOLDS):
            fold_ids = ids[clusters == fold]
            training_samples.loc[training_samples.Id.isin(fold_ids), 'fold'] = fold

    training_samples[['Id', 'Target', 'fold']].to_csv(f'../input/folds_{nb_folds}_emb.csv', index=False)


np.set_printoptions(precision=3, linewidth=200)

split_to_folds_emb()
