
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


# newpath = 'external_data'
# if not os.path.exists(newpath):
#    os.makedirs(newpath)


def get_html(url):
    response = requests.get(url)
    return response.text


# from https://www.proteinatlas.org/download/subcellular_location.tsv.zip
df = pd.DataFrame.from_csv('Christof/assets/subcellular_location.tsv', sep="\t" ,index_col = None)

urls = []
for name in df[['Gene', 'Gene name']].values:
    name = '-'.join(name)
    url = ('https://www.proteinatlas.org/' +name +'/antibody#ICC')
    urls.append(url)

def load_img(url):
    html = get_html(url)
    soup = BeautifulSoup(html, 'lxml')

    links = []
    for a in soup.findAll('a' ,{'class' :'colorbox'} ,href=True):
        if '_selected' in  a['href']:
            links.append(''.join(('https://www.proteinatlas.org' +a['href']).split('_medium')))

    i = 0
    for link in set(links):
        try:
            name = url.split('/')[-2]
            response = requests.get(link)
            img = Image.open(BytesIO(response.content))
            if np.array(img)[: ,: ,0].mean( ) <70:
                img.save('Christof/assets/external_data2/' +name +'_' +str(i ) +'.png')
                i+=1
                # break
        except:
            pass


num_cores = 12
Parallel(n_jobs=num_cores,prefer='threads')(delayed(load_img)(i) for i in urls)
#unsuccessful = []
#for i,url in enumerate(urls):
#    print(i)
#    try:
#        load_img(url)
#    except:
#        unsuccessful += [url]
