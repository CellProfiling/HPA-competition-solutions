import pandas as pd
from os import listdir
from os.path import  join, isfile


LABEL_MAP = {
0: "Nucleoplasm" ,
1: "Nuclear membrane"   ,
2: "Nucleoli"   ,
3: "Nucleoli fibrillar center",
4: "Nuclear speckles"   ,
5: "Nuclear bodies"   ,
6: "Endoplasmic reticulum"   ,
7: "Golgi apparatus"  ,
8: "Peroxisomes"   ,
9:  "Endosomes"   ,
10: "Lysosomes"   ,
11: "Intermediate filaments"  ,
12: "Actin filaments"   ,
13: "Focal adhesion sites"  ,
14: "Microtubules"   ,
15: "Microtubule ends"   ,
16: "Cytokinetic bridge"   ,
17: "Mitotic spindle"  ,
18: "Microtubule organizing center",
19: "Centrosome",
20: "Lipid droplets"   ,
21: "Plasma membrane"  ,
22: "Cell junctions"   ,
23: "Mitochondria"   ,
24: "Aggresome"   ,
25: "Cytosol" ,
26: "Cytoplasmic bodies",
27: "Rods & rings"}

LOC_MAP = {}
for k in LABEL_MAP.keys(): LOC_MAP[LABEL_MAP[k]] = k

dg = pd.read_csv('Christof/assets/subcellular_location.tsv', sep="\t",index_col = None)
dg.set_index('Gene',inplace=True)
print(dg.head())
print(dg.shape)

file_list_x = [f for f in listdir('Christof/assets/train_rgb_512_ext2') if isfile(join('Christof/assets/train_rgb_512_ext2',
                                                                   f))]
print(file_list_x[:15],len(file_list_x))

fid = [f[:-4] for f in file_list_x]
gene = [i[:15] for i in fid]
rel = [dg.loc[g]['Reliability'] for g in gene]

s0 = [str(dg.loc[g]['Enhanced']).split(';') for g in gene]
t0 = [' '.join([str(LOC_MAP[j]) for j in i if j in LOC_MAP]).strip() for i in s0]

s1 = [str(dg.loc[g]['Supported']).split(';') for g in gene]
t1 = [' '.join([str(LOC_MAP[j]) for j in i if j in LOC_MAP]).strip() for i in s1]

s2 = [str(dg.loc[g]['Approved']).split(';') for g in gene]
t2 = [' '.join([str(LOC_MAP[j]) for j in i if j in LOC_MAP]).strip() for i in s2]

s3 = [str(dg.loc[g]['Uncertain']).split(';') for g in gene]
t3 = [' '.join([str(LOC_MAP[j]) for j in i if j in LOC_MAP]).strip() for i in s3]

t = [[y for y in z if len(y) > 0] for z in zip(t0,t1,t2,t3)]
targ = [' '.join(y).strip() for y in t]

print(s0[:20],t0[:20],s1[:20],t1[:20],s2[:20],t2[:20],s3[:20],t3[:20])

dfx = pd.DataFrame({'Id':fid,'Gene':gene,'Reliability':rel,'Target':targ})

print(dfx.shape)

dfx = dfx[dfx['Target'] != '']

print(dfx.head())
print(dfx.shape)

dfx.to_csv('Christof/assets/train_ext2.csv',index=False)