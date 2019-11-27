import os, time
from glob import glob
import pandas as pd
import bs4
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from tqdm import tqdm
import json

# download htmls from HPA website
def getHtmls(f="subcellular_location.tsv", root="htmls"):
    ids = pd.read_table(f)["Gene"].values.tolist()[0:50]
    if not os.path.exists(root):
        os.makedirs(root)
    url_root = "https://www.proteinatlas.org/{id}/cell"
    cmds = []
    for i in ids:
        url = url_root.format(id=i)
        cmd = "wget {url} -O {id}.html".format(url=url, id=root + "/" + i)
        cmds.append(cmd)
    Parallel(n_jobs=8)(delayed(runcmd)(c) for c in tqdm(cmds))

# parser images' url from htmls
def getImgId(html):
    soup = BeautifulSoup(open(html).read(), features="html.parser")
    img_ids = []
    for img in soup.find_all("img"):
        if img["src"].endswith("thumb.jpg") and "base" in img.attrs:
            img_ids.append(img["base"])
    img_ids = list(set(img_ids))
    return img_ids


def runcmd(cmd):
    os.system(cmd)

# download images
def getJpgs(inroot="htmls", outroot="jpgs"):
    if not os.path.exists(outroot):
        os.makedirs(outroot)
    htmls = glob(inroot + "/*.html")
    urlroot = "https://www.proteinatlas.org{img_id}.jpg"
    cmds = []
    for html in tqdm(htmls):
        pid = html.split("/")[-1].split(".html")[0]
        img_ids = getImgId(html)
        for i, img_id in enumerate(img_ids):
            for color in ["blue", "red", "green", "yellow"]:
                print(pid, img_id.split('/')[-2], img_id.split('/')[-1])
                fout = outroot + "/" + pid + '-' + img_id.split('/')[-2] + "-" + img_id.split('/')[-1] + "_" + color + ".jpg"
                print(img_id, fout)
                if os.path.isfile(fout):
                    continue
                cmd = "wget {url} -O {fout}".format(
                    url=urlroot.format(img_id=img_id + "_" + color), fout=fout)
                cmds.append(cmd)


    Parallel(n_jobs=8)(delayed(runcmd)(c) for c in tqdm(cmds))


def rescue():
    f = "test.txt"
    pids = set()
    for line in open(f):
        pid = "-".join(line.split("/")[-1].split("-")[:-1])
        pids.add(pid)
    urlroot = "https://www.proteinatlas.org{img_id}.jpg"
    for pid in pids:
        html = "./htmls/{pid}.html".format(pid=pid)
        img_ids = getImgId(html)
        for i, img_id in enumerate(img_ids):
            for color in ["blue", "green", "red", "yellow"]:
                fout = "./rescue/" + pid + "-" + str(i) + "_" + color + ".jpg"
                if os.path.isfile(fout):
                    continue
                cmd = "wget {url} -O {fout}".format(
                    url=urlroot.format(img_id=img_id + "_" + color), fout=fout)
                # print(cmd)
                os.system(cmd)


if __name__ == '__main__':
    with open('SETTINGS.json', 'r') as f:
        path_dict = json.load(f)
    getHtmls(f=path_dict['EXTERNAL_DATA_DIR'] + "subcellular_location.tsv", root=path_dict['EXTERNAL_DATA_DIR'] + "htmls")
    getJpgs(inroot=path_dict['EXTERNAL_DATA_DIR'] + "htmls", outroot=path_dict['EXTERNAL_DATA_DIR'] + "jpgs")
#rescue()
