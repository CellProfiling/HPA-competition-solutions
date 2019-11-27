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
    return sub

#======================================================================================================================
# Voting
#====================================================================================================================
sub_dir = 'sub_dir_team/'

df_1 =  expand(sub_dir + 'enst31_1shai562_1russ609_1dieter580_1shai593_1dmytro592_kevin599_3.6.csv')
df_2 =  expand('../input/train_test.csv')
df = pd.DataFrame(df_1)
df.to_csv('expand_sub2.csv', index = False)

#=========================
list =[0]
markers = ["s","o", "+", "*", 's', '*']
for i in list:
    x = np.arange(0, 28, 1)
    plt.step(x, column_sum[i] , label=i, marker=markers[i])
plt.legend()
plt.grid(True)
#plt.yscale('log')
plt.show()

sim = df_1*1
sim = sim[:, 1:]
unique, counts = np.unique(sim, return_counts=True)
print (np.asarray((unique, counts)).T)

#===============================
sum = df_1[:, 1:]*1

vote_sub0  = np.where(sum[:,0] >= 4, 1, 0)      #high
vote_sub1  = np.where(sum[:,1] >= 3, 1, 0)
vote_sub2  = np.where(sum[:,2] >= 3, 1, 0)
vote_sub3  = np.where(sum[:,3] >= 3, 1, 0)
vote_sub4  = np.where(sum[:,4] >= 3, 1, 0)
vote_sub5  = np.where(sum[:,5] >= 3, 1, 0)
vote_sub6  = np.where(sum[:,6] >= 3, 1, 0)
vote_sub7  = np.where(sum[:,7] >= 3, 1, 0)
vote_sub8  = np.where(sum[:,8] >= 3, 1, 0)       #low
vote_sub9  = np.where(sum[:,9] >= 3, 1, 0)       #low
vote_sub10 = np.where(sum[:,10] >= 3, 1, 0)      #low
vote_sub11 = np.where(sum[:,11] >= 3, 1, 0)
vote_sub12 = np.where(sum[:,12] >= 3, 1, 0)
vote_sub13 = np.where(sum[:,13] >= 3, 1, 0)
vote_sub14 = np.where(sum[:,14] >= 3, 1, 0)
vote_sub15 = np.where(sum[:,15] >= 2, 1, 0)      # low
vote_sub16 = np.where(sum[:,16] >= 3, 1, 0)
vote_sub17 = np.where(sum[:,17] >= 3, 1, 0)
vote_sub18 = np.where(sum[:,18] >= 3, 1, 0)
vote_sub19 = np.where(sum[:,19] >= 3, 1, 0)
vote_sub20 = np.where(sum[:,20] >= 3, 1, 0)
vote_sub21 = np.where(sum[:,21] >= 3, 1, 0)
vote_sub22 = np.where(sum[:,22] >= 3, 1, 0)
vote_sub23 = np.where(sum[:,23] >= 3, 1, 0)
vote_sub24 = np.where(sum[:,24] >= 3, 1, 0)
vote_sub25 = np.where(sum[:,25] >= 4, 1, 0)      #high
vote_sub26 = np.where(sum[:,26] >= 3, 1, 0)
vote_sub27 = np.where(sum[:,27] >= 3, 1, 0)      #low

vote_sub = np.column_stack((vote_sub0,vote_sub1, vote_sub2, vote_sub3,
                            vote_sub4, vote_sub5,vote_sub6, vote_sub7,
                            vote_sub8, vote_sub9,vote_sub10,vote_sub11,
                            vote_sub12, vote_sub13, vote_sub14, vote_sub15,
                            vote_sub16, vote_sub17, vote_sub18, vote_sub19,
                            vote_sub20,vote_sub21, vote_sub22, vote_sub23,
                            vote_sub24,vote_sub25,vote_sub26, vote_sub27)
                           )
vote_sub_ref = np.where(sum >= 2, 1, 0)

df = pd.DataFrame(vote_sub)
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
submit.to_csv('sub_dir_team/enst29_1shai562_1russ609_1dieter580_1shai592_1dmytro592_kevin602_3.6_highfreq4.6.csv', index=False)


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