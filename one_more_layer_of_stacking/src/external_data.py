import numpy as np
import pandas as pd
import requests
from lxml import etree
from lxml import html

import os
import gc

HPA_TSV_FILE = 'input/proteinatlas.tsv'
TEMP_URLS_FILE = "new_data_urls.csv"
DOWNLOAD_IMAGES_DIR = "input/new_data"

if not os.path.exists(os.path.dirname(TEMP_URLS_FILE)):
    os.makedirs(os.path.dirname(TEMP_URLS_FILE))

if not os.path.exists(DOWNLOAD_IMAGES_DIR):
    os.makedirs(DOWNLOAD_IMAGES_DIR)


protein_atlas = pd.read_csv('input/proteinatlas.tsv', sep='\t')
genes = []
urls = []
labels = []
with open(TEMP_URLS_FILE, 'w') as csv_file:
    for i, row in protein_atlas.iterrows():
        res = requests.get("https://www.proteinatlas.org/{0}.xml".format(row["Ensembl"]))
        tree = etree.fromstring(res.content)

        for data in tree.xpath('//cellExpression/subAssay/data'):
            locations = data.xpath('./location/text()')
            for im_url in data.xpath('./assayImage/image/imageUrl/text()'):
                csv_file.write('{0},{1},{2}\n'.format(row["Ensembl"],
                                                      im_url[35:],
                                                      ";".join(locations)))

image_names = []
parsed_labels = []

for i, row in protein_atlas.iterrows():
    resp = requests.get("https://www.proteinatlas.org/{Ensembl}-{Gene}/cell".format(**row))
    tree = html.fromstring(resp.text)

    images = []
    classes = []
    for tr in tree.xpath('//th[contains(@class, "cellImages")]/table[2]/tbody/tr'):
        bases = tr.xpath('.//img[@base]/@base')
        klass = tr.xpath('.//td[not(@rowspan)][1]/text()')[0]
        if bases:
            for img in images:
                image_names.append(img)
                parsed_labels.append(";".join(classes))

            images = [b for b in bases]
            classes.clear()

        classes.append(klass.lower())
    for img in images:
        image_names.append(img)
        parsed_labels.append(";".join(classes))

new_data_labels = pd.DataFrame()

new_data_labels['Image'] = image_names
new_data_labels['Site Parsed Labels'] = parsed_labels

labels_dict = {
    "Nucleoplasm": 0,
    "Nuclear membrane": 1,
    "Nucleoli": 2,
    "Nucleoli fibrillar center": 3,
    "Nuclear speckles": 4,
    "Nuclear bodies": 5,
    "Endoplasmic reticulum": 6,
    "Golgi apparatus": 7,
    "Peroxisomes": 8,
    "Endosomes": 9,
    "Lysosomes": 10,
    "Intermediate filaments": 11,
    "Actin filaments": 12,
    "Focal adhesion sites": 13,
    "Microtubules": 14,
    "Microtubule ends": 15,
    "Cytokinetic bridge": 16,
    "Mitotic spindle": 17,
    "Microtubule organizing center": 18,
    "Centrosome": 19,
    "Lipid droplets": 20,
    "Plasma membrane": 21,
    "Cell junctions": 22,
    "Mitochondria": 23,
    "Aggresome": 24,
    "Cytosol": 25,
    "Cytoplasmic bodies": 26,
    "Rods & rings": 27,

    # BONUS!
    'Midbody': 16,
    #   'Cleavage furrow':-2,
    'Nucleus': 0,
    #   'Vesicles':-4,
    'Midbody ring': 16
}

labels_dict_lower = dict((k.lower(), i) for k, i in labels_dict.items())

new_data = pd.read_csv(TEMP_URLS_FILE, header=None, names=["Gene", "Url", "XML Labels"])
new_data.drop_duplicates(["Url"], inplace=True)
new_data.index = new_data["Url"].apply(lambda x: "/images/" + x.replace("_blue_red_green.jpg", ""))
new_data.drop("Url", axis=1, inplace=True)

new_data_labels.drop_duplicates(inplace=True)

all_new_data = new_data_labels.join(new_data, on="Image")
all_new_data.fillna("", inplace=True)

for c in ['Site Parsed Labels', "XML Labels"]:
    all_new_data[c] = all_new_data[c].apply(lambda x: " ".join(map(str, sorted(
        set([labels_dict_lower[l.lower()] for l in x.split(';') if l in labels_dict_lower]), reverse=True))))

all_new_data = all_new_data.loc[~(all_new_data['Site Parsed Labels'] == "") | ~(all_new_data['XML Labels'] == "")]

all_new_data["img id"] = all_new_data["Image"].apply(lambda x: x.split("/")[-1])

already_downloaded = pd.DataFrame(os.listdir(DOWNLOAD_IMAGES_DIR), columns=["Path"])
already_downloaded["img id"] = already_downloaded["Path"].apply(lambda x: x[:x.find("_classes")])

for file in already_downloaded.loc[~already_downloaded["img id"].isin(all_new_data["img id"])]["Path"]:
    os.remove(os.path.join(DOWNLOAD_IMAGES_DIR, file))


temp = all_new_data.set_index("img id")
for i, row in already_downloaded.loc[already_downloaded["img id"].isin(all_new_data["img id"])].iterrows():
    new_name = "{0}_classes_{1}.jpg".format(row["img id"],
                                            temp.loc[row["img id"], "Site Parsed Labels"].replace(" ", "_"))
    os.rename(os.path.join(DOWNLOAD_IMAGES_DIR, row["Path"]), os.path.join(DOWNLOAD_IMAGES_DIR, new_name))
del temp

to_download = all_new_data.loc[~all_new_data["img id"].isin(already_downloaded["img id"])]

for i, row in to_download.iterrows():
    url = "https://www.proteinatlas.org{0}_blue_red_green.jpg".format(row['Image'])
    filename = os.path.join(DOWNLOAD_IMAGES_DIR,
                            "{0}_classes_{1}.jpg".format(row["img id"], row["Site Parsed Labels"].replace(" ", "_")))
    try:
        with open(filename, "wb") as f:
            f.write(requests.get(url).content)
    except Exception as e:
        print("ERROR?...", e)
