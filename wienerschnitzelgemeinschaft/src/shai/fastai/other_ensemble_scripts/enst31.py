import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

SAMPLE = '../input/sample_submission.csv'

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

#======================================================================================================================
# Voting
#====================================================================================================================
sub_dir = 'sub_dir_team/'

df_1 =  expand(sub_dir + 'leak_brian_tommy_en_res34swa_re50xt_re101xtswa_wrn_4.8.csv') #0.562
df_2 =  expand(sub_dir + 'Christof_blend_4_580.csv') # 0.580
df_3 =  expand(sub_dir + 'ens76d_609.csv') # 0.609
df_4 =  expand(sub_dir +'enspreds103_12mdl_512-256_wtth0.45_leak.csv') #0.593
df_5 =  expand(sub_dir +'sub37_dpn68_gwap_extra_01_30_592.csv') #0.592
df_6 =  expand(sub_dir +'voted_5_c_599.csv') #0.599

#=========================
list =[0,1,2,3,4,5]
markers = ["s","o", "+", "*", 's', '*']
for i in list:
    x = np.arange(0, 28, 1)
    plt.step(x, column_sum[i] , label=i, marker=markers[i])
plt.legend()
plt.grid(True)
#plt.yscale('log')
plt.show()

sim = df_1*1 + df_2*1 + df_3*1 + df_4*1 + df_5*1 + df_6*1
sim = sim[:, 1:]
unique, counts = np.unique(sim, return_counts=True)
print (np.asarray((unique, counts)).T)

#===============================
sum = df_1[:, 1:]*1 + df_2[:, 1:]*1 + df_3[:, 1:]*1 + df_4[:, 1:]*1 + df_5[:, 1:]*1 ++ df_6[:, 1:]*1

vote_sub = np.where(sum >= 3, 1, 0)

#======================================================================================================================
# prepare submission format
#======================================================================================================================
submit = pd.read_csv(SAMPLE)
prediction = []

for row in tqdm(range(submit.shape[0])):

    str_label = ''

    for col in range(vote_sub.shape[1]):
        if (vote_sub[row, col] < 1):
            str_label += ''
        else:
            str_label += str(col) + ' '
    prediction.append(str_label.strip())

submit['Predicted'] = np.array(prediction)
submit.to_csv('sub_dir_team/enst31_1shai562_1russ609_1dieter580_1shai593_1dmytro592_kevin599_3.6.csv', index=False)


# # view submit distribution
# column_sum = []
# df_1 =  expand(sub_dir + 'leak_brian_tommy_en_res34swa_re50xt_re101xtswa_wrn_4.8.csv') #0.562
# df_2 =  expand(sub_dir + 'ens53d.csv') # 0.590
# df_3 =  expand(sub_dir + 'ens45c.csv') # 0.546
# df_sub =  expand('sub_dir_team/enst1.7_shai562_russ590_russ546_2.3.csv')
# list2 =[0,1,2,3]
# for i in list2:
#     x = np.arange(0, 28, 1)
#     plt.step(x, column_sum[i] , label=i)
# plt.legend()
# plt.grid(True)
# plt.yscale('log')
# plt.show()


print('done')