import pandas as pd
import os

hpa_publi_imgs = pd.read_csv('./data/hpa_public_imgs.csv', index_col=0)
hpa_publi_imgs['group'] = 'hpa_rgby'
# hpa_publi_imgs = pd.read_csv('./data/hpa_public_imgs_extra_only.csv', index_col=0)
# hpa_publi_imgs['folder'] = './data/hpa_public_imgs'
# hpa_publi_imgs['extension'] = 'jpg'

original_train_anno = pd.read_csv('./data/train.csv', index_col=0)
original_train_anno['group'] = 'train_full_size'
# original_train_anno['folder'] = './data/train_full_size_compressed'
# original_train_anno['extension'] = 'jpg'

anno = pd.concat([
    hpa_publi_imgs,
    original_train_anno,
])
anno.to_csv('data/train_with_hpa.csv')
# anno.to_csv('data/train_with_hpa_extra_only.csv')
