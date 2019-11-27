import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

LABEL_MAP = {
    0: 'Nucleoplasm',
    1: 'Nuclear membrane',
    2: 'Nucleoli',
    3: 'Nucleoli fibrillar center',
    4: 'Nuclear speckles',
    5: 'Nuclear bodies',
    6: 'Endoplasmic reticulum',
    7: 'Golgi apparatus',
    8: 'Peroxisomes',
    9: 'Endosomes',
    10: 'Lysosomes',
    11: 'Intermediate filaments',
    12: 'Actin filaments',
    13: 'Focal adhesion sites',
    14: 'Microtubules',
    15: 'Microtubule ends',
    16: 'Cytokinetic bridge',
    17: 'Mitotic spindle',
    18: 'Microtubule organizing center',
    19: 'Centrosome',
    20: 'Lipid droplets',
    21: 'Plasma membrane',
    22: 'Cell junctions',
    23: 'Mitochondria',
    24: 'Aggresome',
    25: 'Cytosol',
    26: 'Cytoplasmic bodies',
    27: 'Rods & rings'}

# computute confusion matrices between two submission files
def f1_confusion(csv0, csv1, num_classes=28):
    c0 = pd.read_csv(csv0)
    c1 = pd.read_csv(csv1)
    assert c0.shape == c1.shape
    s0 = [s if isinstance(s, str) else '' for s in c0.Predicted]
    s1 = [s if isinstance(s, str) else '' for s in c1.Predicted]
    p0 = [s.split() for s in s0]
    p1 = [s.split() for s in s1]
    y0 = np.zeros((c0.shape[0], num_classes)).astype(int)
    y1 = np.zeros((c0.shape[0], num_classes)).astype(int)
    # print(p0[:5])
    for i in range(c0.shape[0]):
        for j in p0[i]: y0[i, int(j)] = 1
        for j in p1[i]: y1[i, int(j)] = 1
    # print(y0[:5])

    y0avg = np.average(y0, axis=0)
    y1avg = np.average(y1, axis=0)
    cm = [confusion_matrix(y0[:, i], y1[:, i]) for i in range(y0.shape[1])]
    fm = [f1_score(y0[:, i], y1[:, i]) for i in range(y0.shape[1])]
    fmas = np.argsort(np.array(fm))
    for j in range(y0.shape[1]):
        i = fmas[j]
        print(i, LABEL_MAP[i])
        print(cm[i], ' %4.2f' % fm[i], ' %6.4f' % y0avg[i], ' %6.4f' % y1avg[i],
              ' %6.4f' % (y0avg[i] - y1avg[i]))
        print()
    #     print('y0avg')
    #     print(y0avg)
    #     print('y1avg')
    #     print(y1avg)
    #     print('y0avg - y1avg')
    #     print(y0avg-y1avg)
    print('f1 macro')
    print(np.mean(fm))
    return f1_score(y0, y1, average='macro')


df_ref =  ('../subs/en12a_original_4.8_leak.csv') #0.638

df_1 =  ('../sub_dir_team/leak_brian_tommy_en_res34swa_re50xt_re101xtswa_wrn_4.8_562.csv')
df_1a =  ('../subs/ens103a_external_leak.csv')
# df_2 =  ( 'sub_dir_team/Christof_blend_4_580.csv')
# df_3 =  ('sub_dir_team/ens85bd_russ_616.csv')
df_4 =  ('../sub_dir_team/enspreds103_12mdl_512-256_wtth0.45_leak_shai_593.csv')
# df_5 =  ('sub_dir_team/hill_m94d_dmytro_627.csv')
# df_6 =  ('sub_dir_team/voted_5_d_kevin_602.csv')
# df_7 =  ('sub_dir_team/hill_b93d_l2_615update.csv')
# df_8 =  ('sub_dir_team/submission_loss_5fold_mean_2_GAP_chrs_602.csv')

# df_6 =  ('sub_dir_team/enstw37_1shai562_3russ616_3dieter580_2shai593_3dmytro617_2kevin602_1l2.615_2dmytro598_8.17_clswt_635.csv')
# df_7 =  ('sub_dir_team/enstw36_1shai562_3russ616_3dieter580_2shai593_3dmytro617_2kevin602_1l2615_7.15_clswt_642.csv')
# df_8 =  ('sub_dir_team/enstw35_1shai562_3russ616_3dieter612_2shai593_3dmytro617_2kevin602_7.14_clswt_640.csv')
#df_9 =  ('sub_dir_team/enst26_1shai562_1russ609_1dieter580_1shai592_1dmytro592_kevin602_3.6_638.csv')
#df_10 =  ('sub_dir_team/enst26_1shai562_1russ609_1dieter580_1shai592_1dmytro592_kevin593_3.6_638.csv')
#df_22 =  ('sub_dir_team/enst31_1shai562_1russ609_1dieter580_1shai593_1dmytro592_kevin599_3.6_636.csv')
#df_11 =  ('sub_dir_team/enstw39b_flat_1sh562_3ru616_2die580_2sh593_3dm627_2ke602_1l2.615updt_2die602_8.16_clswt_642.csv')


#df_12 =  ('sub_dir_team/enstw40_2sh562_2ru616_2die602_2sh593_3dm627_2ke602_1l2.615updt_1die580_1dm598_8.16_clswt.csv')
#df_13 =  ('sub_dir_team/enstw39a_1sh562_3ru616_2die580_2sh593_3dm627_2ke602_1l2.615updt_2die602_8.16_clswt.csv')
#df_14 =  ('sub_dir_team/test.csv')
#df_14 =  ('sub_dir_team/enstw36_1shai562_3russ616_3dieter580_2shai593_3dmytro617_2kevin602_1l2615_7.15_clswt_642.csv')
#df_15 =  ('sub_dir_team/650-enstw43_10en36-642_10en39b-642_10hill2-636_10-646_2.4.csv')
#df_26=  ('sub_dir_team/649-enstw43a_1en36b_1en39b1_1hillu97cd_1hillm95cd_2.4.csv')
#df_36 =  ('sub_dir_team/646-hill_m95cd.csv')
#df_46 =  ('sub_dir_team/643-enstw43_10en36-642_10en39b-642_10hill2-636_10-646_3.4.csv')

#df_1 =  ('sub_dir_team/hill_m94cd_rudy_639.csv') #1 +1
#df_2 =  ('sub_dir_team/hill_u94cd_ru_636.csv') #2 +3
#df_3 =  ('sub_dir_team/enstw36_1shai562_3russ616_3dieter580_2shai593_3dmytro617_2kevin602_1l2615_7.15_clswt_642.csv') # 3 +3
#df_4 =  ('sub_dir_team/enstw39b_flat_1sh562_3ru616_2die580_2sh593_3dm627_2ke602_1l2.615updt_2die602_8.16_clswt_642.csv') # 2+2

#df_111 =  ('sub_dir_team/hill_m95cd_646.csv')
#df_211 =  ('sub_dir_team/hill_m97cd_647.csv')

#df_311 =  ('sub_dir_team/submission_hc_die_629.csv')
#df_411 =  ('sub_dir_team/hill_u97cd_639.csv')  # replaces hill_u94cd_ru_636.csv

#df_511 =  ('sub_dir_team/enstw53.csv')
#df_611 =  ('sub_dir_team/hill_g100cd_653.csv')


f1_confusion(df_ref, df_1)