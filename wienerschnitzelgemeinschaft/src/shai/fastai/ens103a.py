import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

th=0.45

# Manual thresholds
th_m = np.array([th+0.05, th,   th,   th,  th,   th,   th,   th,   th-0.1,   th-0.1,
                 th-0.1,  th,   th,   th,   th,   th-0.15,   th,   th+0.05,   th,   th,
                 th,  th+0.05,  th,   th,   th,   th+0.05,  th,   th-0.2])

base_dir = 'subs/external/'

subs = [#==========================================#
        'res18_512_ext_4chn_5050_3_cyc_0',
        'res18_512_ext_4chn_5050_3_cyc_0-swa',

        'res34_512_ext_4chn_5050_3_cyc_0_cp1',
        'res34_512_ext_4chn_5050_4_cyc_0_cp2',

        'resxt50_512_ext_2_cyc_1_cp1',
        'resxt50_512_ext_3_cyc_0_cp2',
        'resxt50_512_ext_4_cyc_0-swa_cp3',

        'resxt101_512_ext_4chn_5050_2_cyc_1',

        'wrn_512_unbalanced_ext_semi',

        #========================================#
        'res18_256_ext_4chn_5050_3_cyc_0',
        'res18_256_ext_4chn_5050_3_cyc_0-swa',

        'res34_256_ext_4chn_5050_3_cyc_0',
        'res34_256_ext_4chn_5050_3_cyc_0-swa'
        #=========================================#
        ]

SAMPLE = '../input/sample_submission.csv'

all_preds = []
for isub in subs:
    fileName = base_dir + isub + '/' + 'preds.pkl'
    with open(fileName, 'rb') as f:
        a = pkl.load(f)
    a = a[a[:, 0].argsort()]
    logits = a[:, 1:]
    all_preds.append(logits)


sub_n = all_preds[0].astype(np.float32, copy=False)
for i in range (1,len(subs)):
    sub_n += all_preds[i].astype(np.float32, copy=False)
sub_avg = sub_n / len(subs)


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


save_pred(sub_avg, ths = th_m, fname = 'subs/ens103a_external.csv')  # From manual threshold
