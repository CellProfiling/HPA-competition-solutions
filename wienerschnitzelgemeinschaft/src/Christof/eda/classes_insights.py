import pandas as pd
import numpy as np
data = pd.read_csv('Christof/assets/train.csv').sample(frac=0.5,random_state=18)
data['y'] = data['Target'].apply(lambda x: x.split(' '))
data['y'] = data['y'].apply(lambda x: [int(item) for item in x])
for i in range(28):
    data[f'y{i}'] = data['y'].apply(lambda x: int(i in x))



for i in range(28):
    y0_df = data[data[f'y{i}'] == 1]
    columns = [f'y{j}' for j in range(28) if not j == i]
    connections = y0_df[columns].sum().sum()
    perc = connections / len(y0_df)
    print(f'Connections: {i} {connections} perc: {perc}')

y27_df = data[data[f'y{i}'] == 1]

columns = [f'y{j}' for j in range(28)]
numbers = data[columns].sum()