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
df_test =  expand('sub_dir_team/test.csv')

df_1 =  expand('sub_dir_team/hill_g100cd_653.csv')
df_2 =  expand('sub_dir_team/hill_m97cd_647.csv')
df_3 =  expand('sub_dir_team/submission_hc_die_629.csv')
df_4 =  expand('sub_dir_team/hill_u97cd_639.csv')  # replaces hill_u94cd_ru_636.csv

#df_test =  expand('sub_dir_team/enstw42_05en36-642_05en39b-642_10hill1-639_10hill2-636_2.3_642.csv')
#df_test =  expand('sub_dir_team/enstw43_10en36-642_10en39b-642_10hill2-636_10-646_2.4.csv')

# df_1 =  expand('sub_dir_team/650-enstw43_10en36-642_10en39b-642_10hill2-636_10-646_2.4.csv')
# df_2 =  expand('sub_dir_team/649-enstw43a_1en36b_1en39b1_1hillu97cd_1hillm95cd_2.4.csv')
# df_3 =  expand('sub_dir_team/646-hill_m95cd.csv')
# df_4 =  expand('sub_dir_team/643-enstw43_10en36-642_10en39b-642_10hill2-636_10-646_3.4.csv')

# df_1    =  expand('sub_dir_team/hill_m94cd_rudy_639.csv') #1 +1
# df_2    =  expand('sub_dir_team/hill_u94cd_ru_636.csv') #2 +3
# df_3    =  expand('sub_dir_team/enstw36_1shai562_3russ616_3dieter580_2shai593_3dmytro617_2kevin602_1l2615_7.15_clswt_642.csv') # 3 +3
# df_4    =  expand('sub_dir_team/enstw39b_flat_1sh562_3ru616_2die580_2sh593_3dm627_2ke602_1l2.615updt_2die602_8.16_clswt_642.csv') # 2+2

# df_5 =  expand('sub_dir_team/enstw36_1shai562_3russ616_3dieter580_2shai593_3dmytro617_2kevin602_1l2615_7.15_clswt_642.csv')
# df_4 =  expand('sub_dir_team/enstw39a_1sh562_3ru616_2die580_2sh593_3dm627_2ke602_1l2.615updt_2die602_8.16_clswt_640.csv')
# df_3 =  expand('sub_dir_team/enstw39b_flat_1sh562_3ru616_2die580_2sh593_3dm627_2ke602_1l2.615updt_2die602_8.16_clswt_642.csv')
# df_2 =  expand('sub_dir_team/enstw39c_low-2_1sh562_3ru616_2die580_2sh593_3dm627_2ke602_1l2.615updt_2die602_8.16_clswt_631.csv')
# df_1 =  expand('sub_dir_team/enstw39d_low-2_high+2_1sh562_3ru616_2die580_2sh593_3dm627_2ke602_1l2.615updt_2die602_8.16_clswt_630.csv')



# df_2 =  expand('sub_dir_team/enstw40_2sh562_2ru616_2die602_2sh593_3dm627_2ke602_1l2.615updt_1die580_1dm598_8.16_clswt.csv')

#df_4 =  expand('sub_dir_team/enstw37_1shai562_3russ616_3dieter580_2shai593_3dmytro617_2kevin602_1l2.615_2dmytro598_8.17_clswt_635.csv')
# df_5 =  expand('sub_dir_team/enstw39d_low-2_high+2_1sh562_3ru616_2die580_2sh593_3dm627_2ke602_1l2.615updt_2die602_8.16_clswt.csv')
# df_ref =  expand('sub_dir_team/enstw39c_low-2_1sh562_3ru616_2die580_2sh593_3dm627_2ke602_1l2.615updt_2die602_8.16_clswt.csv')
# df_1 =  expand('sub_dir_team/enstw39b_flat_1sh562_3ru616_2die580_2sh593_3dm627_2ke602_1l2.615updt_2die602_8.16_clswt.csv')

#df_6 =  expand('sub_dir_team/enstw42_05en36-642_05en39b-642_10hill1-639_10hill2-636_2.3_642.csv')
#df_3 =  expand('sub_dir_team/enst32_1shai562_2russ616_1dieter580_1shai593_1dmytro592_kevin602_3.7_633.csv')
#df_4 =  expand('sub_dir_team/enst33_1shai562_1russ616_1dieter610_1shai593_1dmytro592_kevin602_3.6_633.csv')



list =[0,1,2,3,4]
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


print('done')