import pandas as pd
from os import listdir
from os.path import  join, isfile
from tqdm import tqdm
data = pd.read_csv('Christof/assets/HPAv18RBGY_wodpl.csv', index_col='Id')
fn = listdir('Christof/assets/ext_tomomi/')


new_data = {}
for f in tqdm(fn):
    #f = fn[i]
    id = '_'.join(f[:-5].split('_')[1:])
    if  id in data.index.values:
        new_data[f] = data.loc[id]['Target']


new_df = pd.DataFrame.from_dict(new_data, orient='index')

new_df = new_df.reset_index()
new_df.columns = ['Id','Target']
new_df.to_csv('train_ext1.csv',index=False)