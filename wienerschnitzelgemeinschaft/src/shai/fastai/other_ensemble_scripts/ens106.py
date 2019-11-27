import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

th=0.41

# Manual thresholds
th_m = np.array([th+0.05, th,   th,   th,  th,   th,   th,   th,   th-0.1,   th-0.1,
                 th-0.1,  th,   th,   th,   th,   th-0.15,   th,   th+0.05,   th,   th,
                 th,  th+0.05,  th,   th,   th,   th+0.05,  th,   th-0.15])

base_dir = 'subs/'

subs = [#==========================================#
        '512/res18_512_ext_4chn_5050_3_cyc_0',
        '512/res18_512_ext_4chn_5050_3_cyc_0-swa',

        '512/res34_512_ext_4chn_5050_3_cyc_0_cp1',
        '512/res34_512_ext_4chn_5050_4_cyc_0_cp2',

        '512/resxt50_512_ext_2_cyc_1_cp1',
        '512/resxt50_512_ext_3_cyc_0_cp2',
        '512/resxt50_512_ext_4_cyc_0-swa_cp3',

        '512/resxt101_512_ext_4chn_5050_2_cyc_1',

        '512/wrn_512_unbalanced_ext_semi',

        #========================================#
        '256/res18_256_ext_4chn_5050_3_cyc_0',
        '256/res18_256_ext_4chn_5050_3_cyc_0-swa',

        '256/res34_256_ext_4chn_5050_3_cyc_0',
        '256/res34_256_ext_4chn_5050_3_cyc_0-swa',
        #=========================================#
        #'dn121'
        #==========================================#
        ]

SAMPLE = '../input/sample_submission.csv'

all_preds = []
for i,isub in enumerate (subs):
    fileName = base_dir + isub + '/' + 'preds.pkl'
    with open(fileName, 'rb') as f:
        a = pkl.load(f)
    a = a[a[:, 0].argsort()]
    logits = a[:, 1:]
    all_preds.append(logits)



sub_n = all_preds[0].astype(np.float32, copy=False)
for i in range (1,len(subs)):
    if i!=13:
        sub_n += all_preds[i].astype(np.float32, copy=False)
    else:
        sub_n += all_preds[i].astype(np.float32, copy=False) * 8

sub_avg = sub_n / 13


fnames = a[:, :1].astype(np.str, copy=False)
c = fnames.tolist()
flattened = [val for sublist in c for val in sublist]

def save_pred(pred, ths=0.5, fname='protein_classification.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line > ths)[0]]))
        pred_list.append(s)

    sample_df = pd.read_csv(SAMPLE)
    sample_list = list(sample_df.Id)
    pred_dic = dict((key, value) for (key, value)
                    in zip(flattened, pred_list))
    pred_list_cor = [pred_dic[id] for id in sample_list]
    df = pd.DataFrame({'Id': sample_list, 'Predicted': pred_list_cor})
    df.to_csv( fname, header=True, index=False)


save_pred(sub_avg, ths = th_m, fname = 'sub_dir_team/ensemble-v1.csv'.format(th))  # From manual threshold

#======================================================================================================================
# show distribution
#======================================================================================================================
label_names = {
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


column_sum = []
sub_name = []

def expand(csv):
    sub = pd.read_csv(csv)
    print(csv, sub.isna().sum())

    sub = sub.replace(pd.np.nan, '0')
    sub[f'target_vec'] = sub['Predicted'].map(lambda x: list(map(int, x.strip().split())))
    for i in range(28):
        sub[f'{label_names[i]}'] = sub['Predicted'].map(
                 lambda x: 1 if str(i) in x.strip().split() else 0)
    sub = sub.values
    sub = np.delete(sub, [1, 2], axis=1)

    a = sub[:, 1:]
    unique, counts = np.unique(a, return_counts=True)
    print('Unique counts:',np.asarray((unique, counts)).T)
    print('Total labels:{} Class-wise:{}'.format(a.sum(), a.sum(axis=0)))
    column_sum.append( a.sum(axis=0))
    sub_name.append(csv)
    return sub

sub_dir = 'subs/'
df_1 =  expand('sub_dir_team/test.csv'.format(th)) #497
df_2 =  expand('sub_dir_team/650-enstw43_10en36-642_10en39b-642_10hill2-636_10-646_2.4.csv')
list =[0,1]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
w=0
for i in list:
    x = np.arange(0, 28, 1)
    plt.bar(x+w, column_sum[i],width = 0.1, color = colors[i], label=sub_name[i], )
    w=w+0.11
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()


