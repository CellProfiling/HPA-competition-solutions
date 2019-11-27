import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import re
import os
from PIL import Image
import requests
from io import BytesIO
from joblib import Parallel, delayed

from glob import glob
from tqdm import tqdm_notebook

from itertools import chain
from collections import Counter
from matplotlib import pyplot as plt


# newpath = 'external_data'
# if not os.path.exists(newpath):
#    os.makedirs(newpath)


def get_html(url):
    response = requests.get(url)
    return response.text


# from https://www.proteinatlas.org/download/subcellular_location.tsv.zip
subcellular_location = pd.read_csv('../input/external-data-for-protein-atlas/subcellular_location.tsv', sep="\t",
                                   index_col=None)

# get urls
urls = []
for name in subcellular_location[['Gene', 'Gene name']].values:
    name = '-'.join(name)
    url = ('https://www.proteinatlas.org/' + name + '/antibody#ICC')
    urls.append(url)


def load_img(url):
    html = get_html(url)
    soup = BeautifulSoup(html, 'lxml')

    links = []
    for a in soup.findAll('a', {'class': 'colorbox'}, href=True):
        if '_selected' in a['href']:
            links.append(''.join(('https://www.proteinatlas.org' + a['href']).split('_medium')))

    i = 0
    for link in set(links):
        try:
            name = url.split('/')[-2]
            response = requests.get(link)
            img = Image.open(BytesIO(response.content))
            if np.array(img)[:, :, 0].mean() < 70:
                img.save('external_data/' + name + '_' + str(i) + '.png')
                i += 1
                # break
        except:
            pass

#===============================================
subcellular_location.head()
#pages with images
urls[:10]
#===============================================

label_names = {
    0:  "Nucleoplasm",
    1:  "Nuclear membrane",
    2:  "Nucleoli",
    3:  "Nucleoli fibrillar center" ,
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",
    6:  "Endoplasmic reticulum",
    7:  "Golgi apparatus",
    8:  "Peroxisomes",
    9:  "Endosomes",
    10:  "Lysosomes",
    11:  "Intermediate filaments",
    12:  "Actin filaments",
    13:  "Focal adhesion sites",
    14:  "Microtubules",
    15:  "Microtubule ends",
    16:  "Cytokinetic bridge",
    17:  "Mitotic spindle",
    18:  "Microtubule organizing center",
    19:  "Centrosome",
    20:  "Lipid droplets",
    21:  "Plasma membrane",
    22:  "Cell junctions",
    23:  "Mitochondria",
    24:  "Aggresome",
    25:  "Cytosol",
    26:  "Cytoplasmic bodies",
    27:  "Rods & rings"
  }

all_label_names = {
    0:  "Nucleoplasm",
    1:  "Nuclear membrane",
    2:  "Nucleoli",
    3:  "Nucleoli fibrillar center" ,
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",
    6:  "Endoplasmic reticulum",
    7:  "Golgi apparatus",
    8:  "Peroxisomes",
    9:  "Endosomes",
    10:  "Lysosomes",
    11:  "Intermediate filaments",
    12:  "Actin filaments",
    13:  "Focal adhesion sites",
    14:  "Microtubules",
    15:  "Microtubule ends",
    16:  "Cytokinetic bridge",
    17:  "Mitotic spindle",
    18:  "Microtubule organizing center",
    19:  "Centrosome",
    20:  "Lipid droplets",
    21:  "Plasma membrane",
    22:  "Cell junctions",
    23:  "Mitochondria",
    24:  "Aggresome",
    25:  "Cytosol",
    26:  "Cytoplasmic bodies",
    27:  "Rods & rings",
    # new classes
    28: "Vesicles",
    29: "Nucleus",
    30: "Midbody",
    31: "Cell Junctions",
    32: "Midbody ring",
    33: "Cleavage furrow"
         }

all_names = []
for j in tqdm_notebook(range(len(subcellular_location))):
    names = np.array(subcellular_location[['Enhanced', 'Supported', 'Approved', 'Uncertain']].values[j])
    names = [name for name in names if str(name) != 'nan']
    split_names = []
    for i in range(len(names)):
        split_names = split_names + (names[i].split(';'))

    all_names.append(split_names)

subcellular_location['names'] = all_names

#===================================

from PIL import Image

img = Image.open(glob('../input/external-data-for-protein-atlas/external_data/*')[0])
print(img.size)
img
#===================================

#only old names
data = []
for i in tqdm_notebook(range(len(subcellular_location))):
    im_name  = subcellular_location['Gene'].values[i]+'-'+subcellular_location['Gene name'].values[i]
    for im in glob('../input/external-data-for-protein-atlas/external_data/'+im_name+'*'):
        labels = []
        for name in subcellular_location['names'].values[i]:
            try:
                if name == 'Rods & Rings': name = "Rods & rings"
                labels.append(list(label_names.values()).index(name))
            except:
                pass
        if len(labels)>0:
            data.append([im.split('/')[-1].split('.png')[0], subcellular_location['names'].values[i], labels])

df = pd.DataFrame(data, columns = ['id', 'names', 'labels'])
df.head()

count_labels = Counter(list(chain.from_iterable(df['labels'].values)))
plt.figure(figsize = (16,10))
plt.bar(list(label_names), [count_labels[k] for k in list(label_names)],)
plt.xticks(list(label_names),list(label_names.values()), rotation=90, size = 15)

for i in count_labels:
    plt.text(i-0.4,count_labels[i], count_labels[i], size =12)

plt.show()