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


# # view submit distribution

df_ref = expand('sub_dir_team/enstw36_1shai562_3russ616_3dieter580_2shai593_3dmytro617_2kevin602_1l2615_7.15_clswt_642.csv') #0.640
df_1 =  expand('sub_dir_team/leak_brian_tommy_en_res34swa_re50xt_re101xtswa_wrn_4.8_562.csv')
df_2 =  expand( 'sub_dir_team/Christof_blend_4_580.csv')
df_3 =  expand('sub_dir_team/ens85bd_russ_616.csv')
df_4 =  expand('sub_dir_team/enspreds103_12mdl_512-256_wtth0.45_leak_shai_593.csv')
df_5 =  expand('sub_dir_team/hill_m94d_dmytro_627.csv')
df_6 =  expand('sub_dir_team/voted_5_d_kevin_602.csv')
df_7 =  expand('sub_dir_team/hill_b93d_l2_615update.csv')
df_8 =  expand('sub_dir_team/submission_loss_5fold_mean_2_GAP_chrs_602.csv')

list =[0,1,2,3,4,5,6,7,8]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tab:brown', 'orange']
w=0
for i in list:
    x = np.arange(0, 28, 1)
    plt.bar(x+w, column_sum[i],width = 0.07, color = colors[i], label=sub_name[i], )
    w=w+0.08
plt.legend()
plt.grid(True)
#plt.yscale('log')
plt.show()

# sim = df_1*1 + df_2*1 + df_3*1 + df_4*1 + df_5*1 + df_6*1 + df_ref*1
# sim = sim[:, 1:]
# unique, counts = np.unique(sim, return_counts=True)
# print (np.asarray((unique, counts)).T)

print('done')