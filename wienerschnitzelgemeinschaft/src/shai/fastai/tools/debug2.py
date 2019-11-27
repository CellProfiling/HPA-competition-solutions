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

df_7 =  expand('sub_dir_team/enstw35_1shai562_3russ616_3dieter612_2shai593_3dmytro617_2kevin602_7.14_clswt_640.csv')
df_1 =  expand('sub_dir_team/leak_brian_tommy_en_res34swa_re50xt_re101xtswa_wrn_4.8_562.csv') #0.562
df_2 =  expand( 'sub_dir_team/submission_blend1_dieter_612.csv') # 0.612
df_3 =  expand('sub_dir_team/ens85bd_russ_616.csv') # 0.616
df_4 =  expand('sub_dir_team/enspreds103_12mdl_512-256_wtth0.45_leak_shai_593.csv') #0.593
df_5 =  expand('sub_dir_team/hill_m92d_dmytro_617.csv') #0.617
df_6 =  expand('sub_dir_team/voted_5_d_kevin_602.csv') #0.602

#=======================================================================================================================
# Show distribution
#=======================================================================================================================
list =[0,1,2,3,4,5, 6]
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

sim = df_1*1 + df_2*1 + df_3*2 + df_4*1 + df_5*1 + df_6*1
sim = sim[:, 1:]
unique, counts = np.unique(sim, return_counts=True)

#======================================================================================================================
# voting
#====================================================================================================================

sum = df_1[:, 1:]*1 + df_2[:, 1:]*1 + df_3[:, 1:]*1 + df_4[:, 1:]*1 + df_5[:, 1:]*1 ++ df_6[:, 1:]*1

vote_sub = np.where(sum >= 1, 1, 0)

sname = df_2[:, 0]
vote_sub_2 = np.column_stack((sname,vote_sub))
df = pd.DataFrame(vote_sub_2)
df.to_csv('vote_sub.csv', index = False)
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
submit.to_csv('sub_dir_team/enst33_1shai562_1russ616_1dieter612_1shai593_1dmytro617_kevin602_1.6.csv', index=False)


print('done')