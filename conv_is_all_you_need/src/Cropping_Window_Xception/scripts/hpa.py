import glob
import os
import re
import shutil
from itertools import chain
import xml.etree.ElementTree as ET
import xml.sax
from itertools import islice
from multiprocessing import Pool
from urllib.request import urlretrieve
from xml.sax.handler import ContentHandler

import fire
import pandas as pd

from lib.utils import debug, info, pjoin, multiprocessing, str_to_labels

download_folder = 'data/hpa_public_imgs_rgby'
os.makedirs(download_folder, exist_ok=True)
tmp_folder = './tmp/hpa_public_imgs'
os.makedirs(tmp_folder, exist_ok=True)
regex_id_from_img_url = re.compile(r'http:\/\/v18\.proteinatlas\.org\/images\/([0-9]+)\/(.*)_blue_red_green\.jpg')


def download_one(img_id, img_url_match):
    download_report = {}
    errors = []
    for channel in ['red', 'green', 'blue', 'yellow']:
        img_url_single_channel = f"http://v18.proteinatlas.org/images/{img_url_match[1]}/{img_url_match[2]}_{channel}.jpg"
        saved_filename = f'{img_id}_{channel}.jpg'
        saved_path = pjoin(download_folder, saved_filename)
        tmp_path = pjoin(tmp_folder, saved_filename)
        if os.path.exists(saved_path):
            file_size = os.stat(saved_path).st_size
            download_report[channel] = f"cache_found ({file_size})"
        else:
            try:
                urlretrieve(img_url_single_channel, tmp_path)
                file_size = os.stat(tmp_path).st_size
                shutil.move(tmp_path, saved_path)
                download_report[channel] = f"downloaded ({file_size})"
            except Exception as e:
                error_msg = {
                    'id_': img_id,
                    'message': repr(e),
                    'channel': channel,
                }
                debug(error_msg)
                download_report[channel] = f"failed ({repr(e)})"

    return download_report


def download_all_images(n_threads=70):
    xml_fps = glob.glob('./data/protein_xmls/*.xml')
    result_rows = []
    for result in multiprocessing(download_one, ({'xml_fp': fp} for fp in xml_fps), len_=len(xml_fps)):
        if len(result['errors']) > 0:
            result_rows.append(result)

    error_df = pd.DataFrame.from_records(chain(*[row['errors'] for row in result_rows]))
    print(error_df)
    error_df.to_csv('tmp/error_df.csv', index=False)

    info('ALL DONE !!')


def create_anno_for_one_xml(task):
    xml_fp = task['xml_fp']
    et = ET.parse(xml_fp)
    root = et.getroot()
    name = root.find('./name').text
    ensembl_id = root.find('./identifier').attrib['id']
    rows = []
    for antibody_el in root.findall('./antibody'):
        for cell_expression_el in antibody_el.findall('./cellExpression'):
            for subassay_el in cell_expression_el.findall('./subAssay'):
                verification = ' | '.join([x.text for x in subassay_el.findall('./verification')])
                for data_el in subassay_el.findall('./data'):
                    cell_line = data_el.find('./cellLine').text
                    location = ' | '.join([x.text for x in data_el.findall('./location')])
                    assay_image_el = data_el.find('./assayImage')
                    for image_ in assay_image_el.findall('./image'):
                        image_url = image_.find('./imageUrl').text
                        img_url_match = regex_id_from_img_url.match(image_url)
                        img_id = f'{img_url_match[1]}_{img_url_match[2]}'
                        download_report = download_one(img_id, img_url_match)
                        rows.append(
                            {
                                'ensembl_id': ensembl_id,
                                'name': name,
                                'cell_line': cell_line,
                                'location': location,
                                'verification': verification,
                                'img_url': image_url,
                                'img_id': img_id,
                                **download_report,
                            }
                        )
    df = pd.DataFrame.from_records(rows)
    return {
        'task': task,
        'df': df,
    }


def create_anno(n_threads=70):
    xml_fps = glob.glob('./data/protein_xmls/*.xml')
    create_anno_list = [{'xml_fp': xml_fp} for xml_fp in xml_fps]
    with Pool(n_threads) as p:
        result_iter = p.imap_unordered(create_anno_for_one_xml, create_anno_list)
        df_list = []
        for i_result, result in enumerate(result_iter):
            info(f"({i_result}/{len(create_anno_list)}) {result['task']['xml_fp']}")
            df_list.append(result['df'])
    df = pd.concat(df_list, ignore_index=True)
    df = df[[
        'img_id',
        'ensembl_id',
        'name',
        'location',
        'verification',
        'cell_line',
        'red',
        'green',
        'blue',
        'yellow',
        'img_url',
    ]]
    df.to_csv('./data/hpa_public_imgs_meta.csv', index=False)


extra_classes = []

class_name_to_idx = {
    'cleavage furrow': 16,
    'midbody': 16,
    'midbody ring': 16,
    'nucleus': 0,
    'vesicles': -1,
    # ----- above are extra classes ----
    'nucleoplasm': 0,
    'nuclear membrane': 1,
    'nucleoli': 2,
    'nucleoli fibrillar center': 3,
    'nuclear speckles': 4,
    'nuclear bodies': 5,
    'endoplasmic reticulum': 6,
    'golgi apparatus': 7,
    'peroxisomes': 8,
    'endosomes': 9,
    'lysosomes': 10,
    'intermediate filaments': 11,
    'actin filaments': 12,
    'focal adhesion sites': 13,
    'microtubules': 14,
    'microtubule ends': 15,
    'cytokinetic bridge': 16,
    'mitotic spindle': 17,
    'microtubule organizing center': 18,
    'centrosome': 19,
    'lipid droplets': 20,
    'plasma membrane': 21,
    'cell junctions': 22,
    'mitochondria': 23,
    'aggresome': 24,
    'cytosol': 25,
    'cytoplasmic bodies': 26,
    'rods & rings': 27
}


def location_str_to_labels(str_, delimiter=' | ', sort=False):
    if type(str_) is not str:
        return 'ERROR EMPTY'
    elif str_ == '':
        return 'ERROR EMPTY'
    else:
        xs = [class_name_to_idx.get(x) for x in str_.split(delimiter)]
        if any([x < 0 for x in xs]):
            return 'ERROR UNKNOWN CLASS'
        return ' '.join([str(x) for x in xs])


# yapf: disable
rare_classes = [
     #                    organelle  n_samples
27 , #                 Rods & rings         11
15 , #             Microtubule ends         21
10 , #                    Lysosomes         28
9  , #                    Endosomes         45
8  , #                  Peroxisomes         53
20 , #               Lipid droplets        172
17 , #              Mitotic spindle        210
24 , #                    Aggresome        322
26 , #           Cytoplasmic bodies        328
16 , #           Cytokinetic bridge        530
13 , #         Focal adhesion sites        537
12 , #              Actin filaments        688
22 , #               Cell junctions        802
18 , #Microtubule organizing center        902
6  , #        Endoplasmic reticulum       1008
14 , #                 Microtubules       1066
11 , #       Intermediate filaments       1093
# 1  , #             Nuclear membrane       1254
# 19 , #                   Centrosome       1482
# 3  , #    Nucleoli fibrillar center       1561
# 4  , #             Nuclear speckles       1858
# 5  , #               Nuclear bodies       2513
# 7  , #              Golgi apparatus       2822
# 23 , #                 Mitochondria       2965
# 2  , #                     Nucleoli       3621
# 21 , #              Plasma membrane       3777
# 25 , #                      Cytosol       8228
# 0  , #                  Nucleoplasm      12885
]
# yapf: enabled


def row_has_rare_class(row):
    labels = str_to_labels(row['Target'])
    return any([x in labels for x in rare_classes])


def clean_and_filter_meta():
    df = pd.read_csv('./data/hpa_public_imgs_meta.csv')
    df = df.loc[df.apply(
        lambda row: all(['failed' not in row[channel] for channel in ['red', 'green', 'blue', 'red']]), axis=1
    )]
    anno = pd.DataFrame()
    anno['Id'] = df['img_id']
    anno['Target'] = list(set([location_str_to_labels(x) for x in df['location']]))
    anno = anno[anno.apply(lambda row: len(row['Target']) > 0, axis=1)]
    anno = anno[anno.apply(lambda row: 'ERROR' not in row['Target'], axis=1)]
    anno = anno[anno.apply(row_has_rare_class, axis=1)]
    anno = anno.groupby('Id').first().reset_index()
    anno.to_csv('./data/hpa_public_imgs.csv', index=False)


def combine_hpa_with_train():
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


if __name__ == '__main__':
    # create_anno()
    clean_and_filter_meta()
    combine_hpa_with_train()

# for data_item in data_list:
#     cell_line = data_item.find('./cellLine').text
#     location = data_item.find('./location').text
#     data_item.find_all()

# with open('./data/proteinatlas.xml', 'r') as f:
#     parser = xml.sax.make_parser()
#     parser.setContentHandler(MyContentHandler())
#     parser.parse(f)
