import os
ope = os.path.exists
import numpy as np
import socket
import warnings
warnings.filterwarnings('ignore')

sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
hostname = socket.gethostname()
print('run on %s' % hostname)

RESULT_DIR     = "/data4/data/protein/result"
DATA_DIR       = "/data5/data/protein"
PRETRAINED_DIR = "/data5/data/pretrained"
TIF_DIR        = "/data2/data/protein"
EXTERNEL_DIR   = "/data/data/protein"

PI  = np.pi
INF = np.inf
EPS = 1e-12

IMG_SIZE      = 512
NUM_CLASSES   = 28
ID            = 'Id'
PREDICTED     = 'Predicted'
TARGET        = 'Target'
PARENTID      = 'ParentId'
EXTERNAL      = 'External'
ANTIBODY      = 'antibody'
ANTIBODY_CODE = 'antibody_code'

LABEL_NAMES = {
    0:  "Nucleoplasm",
    1:  "Nuclear membrane",
    2:  "Nucleoli",
    3:  "Nucleoli fibrillar center",
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
LABEL_NAME_LIST = [LABEL_NAMES[idx] for idx in range(len(LABEL_NAMES))]

COLOR_INDEXS = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'yellow': 0,
}
