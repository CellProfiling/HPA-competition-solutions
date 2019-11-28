def sf_stack():
    import importlib
    import sf_utils; importlib.reload(sf_utils)
    from sf_utils import *

    test_meta = pd.read_csv('input/test_meta.csv')
    test_meta.shape

    train_val_meta = pd.read_csv('input/train_meta.csv')
    new_data_meta = pd.read_csv('input/new_data_meta.csv')
    new_data_meta.columns = train_val_meta.columns
    val_meta = pd.concat([new_data_meta, train_val_meta], sort=False)
    val_meta.shape

    val_meta.head()
    model_stats = imagenet_stats

    import pretrainedmodels
    import pretrainedmodels.utils as pqutils
    #print(pretrainedmodels.__dict__.keys())
    _model_name = 'se_resnext50_32x4d'
    model = pretrainedmodels.__dict__[_model_name](num_classes=1000, pretrained='imagenet')
    tf_img = pqutils.TransformImage(model)
    model_stats = (tf_img.__dict__['mean'], tf_img.__dict__['std'])
    model_stats


    data_dir = 'input/'
    valid_df = pd.read_csv('input/' + 'val_id.csv', header=None, names=['idx','Id'])
    train_df = pd.read_csv(data_dir + 'train.csv')
    len(train_df)


    from PIL import Image as QImage
    ids = []
    labels = []
    def file_jpg_to_png(path):
        global ids
        gclasses = set(list(range(28))) - set([0,25])
        f1 = 'input/new_data/' + path + '.jpg'
        f2 = 'input/train_png/' + path + '.png'
        xs = path.split('_')
        q = xs.index('classes') + 1
        xs = xs[q:]
        if len(gclasses & set([int(x) for x in xs])) == 0:
            return
        xs = ' '.join(xs)
        if not os.path.isfile(f2):
            try:
                im = QImage.open(f1)
                im = im.resize((512, 512), QImage.NEAREST)
                im.save(f2)
                ids.append(path)
                labels.append(xs)
            except:
                pass
        else:
            ids.append(path)
            labels.append(xs)

    need_to_prepare_extra = False
    if need_to_prepare_extra:
        for filename in tqdm_notebook(os.listdir('input/new_data/'), total = 68628):
            if filename.endswith(".jpg"):
                file_jpg_to_png(filename[:-4])


    if need_to_prepare_extra:
        data_dir = 'input/'
        xtra_data = pd.DataFrame()
        xtra_data['Id'] = ids
        xtra_data['Target'] = labels
        xtra_data.to_csv(data_dir + 'xtra_train.csv', index=False)
        xtra_data.head(n=3)


    test_matches = pd.read_csv('test_matches.csv')
    test_matches.Extra = test_matches.Extra.apply(lambda x : "_".join(x.split("_")[2:]))
    test_matches.head()

    xtra_data = pd.read_csv(data_dir + 'xtra_train.csv')
    xtra_data['Extra'] = xtra_data.Id.apply(lambda x : x[:x.find("_classes")])
    xtra_data.head()

    xtra_matches_ids = test_matches.Extra.values.tolist()

    xtra_data_train = xtra_data.loc[~xtra_data.Extra.isin(xtra_matches_ids),['Id','Target']].reset_index(drop=True)
    xtra_data_valid = xtra_data.loc[xtra_data.Extra.isin(xtra_matches_ids),['Id','Target']].reset_index(drop=True)


    data = xtra_data_train
    labels = np.zeros((data.shape[0], 28), dtype=np.int32)
    if "Target" in data:
        for i, lbls in data['Target'].str.split().iteritems():
            for j in map(int, lbls):
                labels[i, j] = 1
    for j in range(28):
        print(j,'\t',name_label_dict[j], '\t', labels[:,j].sum(), '\t', labels[:,j].sum()/labels.shape[0])

    xtra_matches_ids = ['1054_E4_1_classes_25_16_0','1762_G4_5_classes_27','1335_C6_2_classes_3',
                        '935_D5_2_classes_22_0','27_H9_2_classes_10','669_D8_1_classes_16_2',
                        '1178_D4_2_classes_19_16_14','791_A9_1_classes_10_9','759_F9_9_classes_25_21_19_16',
                        '1283_F10_2_classes_16_0','688_E7_10_classes_23','1772_F9_7_classes_25_17',
                        '454_E5_1_classes_14_0','1020_C5_3_classes_23','1386_G4_2_classes_8',
                        '681_G8_5_classes_13','1609_C4_2_classes_16_0','690_D3_5_classes_22_21_1_0',
                        '1245_B2_2_classes_21_0','1335_C10_4_classes_16_0','693_A11_3_classes_23',
                        '1139_A12_4_classes_23','916_F8_1_classes_25_2_0','694_C1_2_classes_18_1',
                        '929_B8_1_classes_25_19','340_F5_3_classes_13','138_B12_1_classes_8',
                        '932_G11_2_classes_25_16','28_H9_1_classes_10','924_F12_1_classes_27',
                        '682_F12_2_classes_25_4','1147_D3_13_classes_16_0','346_A5_1_classes_12',
                        '616_F1_4_classes_8','73_A10_1_classes_27_25','663_A9_2_classes_16_14',
                        '859_C8_4_classes_16_14','933_C10_4_classes_22_21','1207_B10_7_classes_12',
                        '694_F10_1_classes_25_21','908_E3_1_classes_4','1758_C9_4_classes_17_2',
                        '1335_D2_2_classes_2_0','929_H2_2_classes_23','1717_G8_34_classes_25_17',
                        '1150_H4_7_classes_13','1054_E4_2_classes_25_16_0','504_B1_3_classes_25_16_0',
                        '747_B5_4_classes_10_9','1020_B1_7_classes_23_5','918_H10_2_classes_25_15',
                        '532_H3_1_classes_25_16_0','757_C6_3_classes_16_2','1346_H6_3_classes_16_5_0','496_D1_1_classes_16_0','1042_C3_3_classes_27','929_B12_1_classes_3','684_C4_2_classes_23_0','696_C9_5_classes_25_21_0','1144_A10_4_classes_2','846_A8_2_classes_16_14','903_F12_2_classes_23_5','1264_G1_1_classes_27','925_H8_2_classes_1_0','121_C6_2_classes_10_9','1657_E10_3_classes_25_17','932_G11_1_classes_25_16','704_G4_1_classes_25_12','1039_C3_2_classes_19_16','906_H7_2_classes_25_6','19_H7_2_classes_8','725_G10_2_classes_16_14','681_B2_4_classes_4','697_A6_4_classes_19_0','1581_B12_2_classes_16_14','926_F7_2_classes_5_0','1770_D2_4_classes_21_17_4','1037_F4_3_classes_19','1413_F11_6_classes_21_16','694_A2_1_classes_2','1049_D11_2_classes_25_16_0','1276_C3_2_classes_21_0','346_B12_3_classes_14_0','1773_G12_3_classes_16_12','1183_F4_2_classes_15','1158_H11_8_classes_16_5','380_C6_1_classes_16_0','792_B6_7_classes_13_0','682_C9_6_classes_25_12_2','906_A9_4_classes_20_0','400_D3_2_classes_25_7','1237_G1_4_classes_21_6','793_B1_1_classes_25_22_0','1308_A5_4_classes_5','800_E1_1_classes_16_14','1421_G5_7_classes_17','906_A9_6_classes_20_0','1245_B2_3_classes_21_0','626_D7_6_classes_25_21_12','344_G2_4_classes_11','901_E12_1_classes_25_6_2','1050_F6_6_classes_16_0','240_G8_1_classes_8','933_C2_1_classes_23_2_0','556_B9_1_classes_25_18_0','1335_C10_2_classes_16_0','1125_F6_3_classes_4','1495_F7_3_classes_7_0','694_C1_1_classes_18_1','918_B3_4_classes_14','1762_E6_5_classes_7','915_C6_5_classes_4','820_G4_3_classes_10_9','927_F12_12_classes_18_0','901_D10_2_classes_12_0','1642_G7_34_classes_25_16','928_G1_2_classes_14_7','682_G9_1_classes_7_0','903_F2_1_classes_2_0','1645_E1_32_classes_16_14','685_G10_5_classes_12_0','927_A9_10_classes_25_5','957_G6_4_classes_16','757_C6_2_classes_16_2','1213_C4_2_classes_4','909_A6_1_classes_2','694_D6_2_classes_1_0','480_D6_3_classes_25_16','1050_F1_3_classes_25_16_0','692_A1_5_classes_25_14_0','1772_H1_5_classes_18_17_16_0','991_G6_7_classes_10_9','782_F8_2_classes_25_16','693_H4_1_classes_7','1259_A11_4_classes_19_16','1414_D12_2_classes_21_0','1139_D5_5_classes_5','930_H3_2_classes_1','901_G9_5_classes_25_19_0','1754_G2_34_classes_5','353_A9_1_classes_21_13','1179_H7_1_classes_25_16_0','1423_A4_2_classes_16_14','686_F4_2_classes_22_21','1693_E1_2_classes_23_16','400_H8_2_classes_23','1680_G4_4_classes_16','935_G3_1_classes_5','838_E8_1_classes_3','1030_D8_2_classes_7_0','684_D12_4_classes_18','812_C10_2_classes_13_0','1416_D10_6_classes_21_16_0','1293_E3_2_classes_1_0','480_D6_2_classes_25_16','700_H6_2_classes_25_2','1773_E10_4_classes_16_0','611_E10_1_classes_25_13','346_B12_4_classes_14_0','523_A9_4_classes_5','1581_B12_3_classes_16_14','684_D8_6_classes_25_12_0','927_F12_11_classes_18_0','353_E4_2_classes_5','556_C1_5_classes_25_22_16','1179_H7_2_classes_25_16_0','1711_B12_3_classes_26_21_4','449_G8_2_classes_4_2','544_A8_5_classes_22_21_7','1772_H1_3_classes_18_17_16_0','1772_G2_6_classes_25_19_16_0','909_C11_2_classes_2_0','930_C12_1_classes_18_14_6','690_C10_2_classes_13','1009_B6_2_classes_10_9','757_E10_5_classes_12','88_D7_2_classes_8','383_E8_7_classes_25_17','1432_F2_2_classes_6','505_C10_1_classes_25_15','1104_E7_2_classes_16_14','699_E8_1_classes_1','1213_C4_3_classes_4','690_H5_1_classes_4','1169_D3_6_classes_16_0','686_F4_1_classes_22_21','532_D1_1_classes_16_0','896_G8_3_classes_5_0','934_G4_3_classes_21','344_G2_1_classes_11','369_C9_1_classes_18_14_0','682_F12_1_classes_25_4','683_E1_2_classes_25_1_0','697_G3_6_classes_13_7','1772_A6_7_classes_5','933_C4_6_classes_5','1231_F9_5_classes_7','802_D5_9_classes_16_0','682_G10_1_classes_7','850_C1_9_classes_21_0','929_B12_2_classes_3','1339_D3_3_classes_2_1','858_D4_2_classes_4','334_B12_2_classes_4','622_F1_7_classes_8','908_G5_2_classes_2_0','778_G6_2_classes_25_16_14','1027_C4_1_classes_7','886_C10_5_classes_23_0','807_C2_3_classes_4','1314_D2_2_classes_25_16_0','1770_B5_1_classes_21_16_11','1105_F10_2_classes_16_0','1283_B2_10_classes_16_0','583_E11_1_classes_25_16','820_G4_7_classes_10_9','928_H3_2_classes_14_0','970_H1_4_classes_25_18','1751_A7_32_classes_27','701_H10_2_classes_25_14','1773_B6_11_classes_23_17_16','1736_G7_31_classes_25_16','928_H3_1_classes_14_0','1645_E5_34_classes_17','539_B3_1_classes_25_21_0','683_E1_1_classes_25_1_0','484_G6_3_classes_22','928_A1_1_classes_4','1773_B6_7_classes_23_17_16','1255_A3_4_classes_16_0','698_C6_2_classes_25_21_4','1773_D5_6_classes_17','681_G8_4_classes_13','935_H11_2_classes_22_0','1125_B9_4_classes_25_7','698_F11_1_classes_13_0','344_F7_1_classes_25_21','906_C11_1_classes_4','1656_F5_2_classes_19_17','1761_A10_3_classes_23_17_14','1772_H5_7_classes_17_7','910_B8_1_classes_12_0','1283_F10_4_classes_16_0','508_C10_1_classes_25_15','681_B2_3_classes_4','868_E8_2_classes_17_16_0','1339_B9_2_classes_16_0','856_A2_4_classes_2_0','700_C3_6_classes_21','869_B3_1_classes_16_0','701_B9_2_classes_21_13_0','1178_F9_6_classes_16_0','542_G1_1_classes_11_2_0']

    xtra_data_train = xtra_data.loc[~xtra_data.Id.isin(xtra_matches_ids),['Id','Target']].reset_index(drop=True)
    xtra_data_valid = xtra_data.loc[xtra_data.Id.isin(xtra_matches_ids),['Id','Target']].reset_index(drop=True)
    xtra_data_train.shape

    exclude_valid = ['5ae3db3a-bbc4-11e8-b2bc-ac1f6b6435d0',
     'e6d0b648-bbbc-11e8-b2ba-ac1f6b6435d0',
     '3202385a-bbca-11e8-b2bc-ac1f6b6435d0',
     '0cf36c82-bbca-11e8-b2bc-ac1f6b6435d0',
     '7cb0006e-bbaf-11e8-b2ba-ac1f6b6435d0',
     '87b77dd2-bba2-11e8-b2b9-ac1f6b6435d0',
     '62c88efa-bbc8-11e8-b2bc-ac1f6b6435d0',
     '44d819c2-bbbb-11e8-b2ba-ac1f6b6435d0',
     'b1ca2b40-bbbd-11e8-b2ba-ac1f6b6435d0',
     '8cd67266-bbbe-11e8-b2ba-ac1f6b6435d0',
     'cead83ec-bb9a-11e8-b2b9-ac1f6b6435d0',
     'a166d11a-bbca-11e8-b2bc-ac1f6b6435d0',
     '91a0a67e-bb9e-11e8-b2b9-ac1f6b6435d0',
     '2be24582-bbb1-11e8-b2ba-ac1f6b6435d0']
    exclude_train = ['7138c4aa-bb9b-11e8-b2b9-ac1f6b6435d0',
     '8a10533e-bba6-11e8-b2ba-ac1f6b6435d0',
     'be92e108-bbb5-11e8-b2ba-ac1f6b6435d0',
     'abfa727e-bba4-11e8-b2ba-ac1f6b6435d0',
     '2384acac-bbae-11e8-b2ba-ac1f6b6435d0',
     'c7a7a462-bbb1-11e8-b2ba-ac1f6b6435d0',
     '559f7ce0-bbb2-11e8-b2ba-ac1f6b6435d0']

    rem_val = pd.read_csv('input/remove_from_val.csv',header=None)
    rem_val.columns = ['Id']
    rem_val.head()

    data_dir = 'input/'
    valid_df = pd.read_csv('input/' + 'val_id.csv', header=None, names=['idx','Id'])
    valid_df = valid_df.loc[~valid_df.Id.isin(exclude_valid),:]
    train_df = pd.read_csv(data_dir + 'train.csv')
    train_df = train_df.loc[~train_df.Id.isin(exclude_train),:]

    test_df = pd.read_csv('input/' + "sample_submission.csv")
    train = train_df.loc[~train_df.Id.isin(valid_df.Id.values.tolist()),:].reset_index(drop=True)
    train = pd.concat([train,xtra_data_train], axis=0, sort=False)
    valid = train_df.loc[train_df.Id.isin(valid_df.Id.values.tolist()),:].reset_index(drop=True)
    valid = pd.concat([valid,xtra_data_valid], axis=0, sort=False)
    #valid = valid.loc[~valid.Id.isin(rem_val.Id.values),:].reset_index(drop=True)
    test  = test_df
    del train_df,valid_df,test_df,xtra_data_valid,xtra_data_train
    gc.collect()


    set(valid.Id.apply(lambda x : x.split('_class')[0]).unique().tolist()) - set(val_meta.Id.unique().tolist())
    set(test.Id.values.tolist()) - set(test_meta.Id.values.tolist())

    qqdata = valid.copy()
    labels = np.zeros((qqdata.shape[0], 28), dtype=np.int32)
    if "Target" in qqdata:
        for i, lbls in qqdata['Target'].str.split().iteritems():
            for j in map(int, lbls):
                labels[i, j] = 1
    for j in range(28):
        print(j,'\t',name_label_dict[j], '\t', labels[:,j].sum(), '\t', labels[:,j].sum()/labels.shape[0])

    def twenty_kfold_threshold(y_true, y_pred):
        n_classes = len(name_label_dict)
        classes_thresholds = []
        classes_scores = []
        for i in range(n_classes):
            for j in range(20):
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=239 + j*101)
                kf_class_thresholds = []
                for _, tst_inx in kf.split(y_true,y_true[:,i]):
                    t_min = np.min(y_pred[tst_inx,i])
                    t_max = np.max(y_pred[tst_inx,i])
                    thresholds = np.linspace(t_min, t_max, 100)
                    scores = np.array([
                        f1_score(y_true[tst_inx,i], np.int32(y_pred[tst_inx,i] >= threshold)) for threshold in thresholds
                    ])
                    threshold_best_index = np.argmax(scores)
                    kf_class_thresholds.append(thresholds[threshold_best_index])
            threshold = np.mean(kf_class_thresholds)
            classes_thresholds.append(threshold)
            f1 = f1_score(y_true[:,i], np.int32(y_pred[:,i] >= threshold))
            classes_scores.append(f1)
        return classes_thresholds, classes_scores

    models = ['sf_fastai_inf_seresnext','sf_fastai_inf_incv4','pd_seresnext_v5','sf_fastai_inf_sev2',
              'db_seresnext_v3_LSEP4','pd_seresnext_v6','pd_inceptionv4_v2',
              'pd_seresnext_v7','pd_xception_v1', 'pd_seresnext_v10']
    results_dir = 'results/'

    def load_model(m):
        ids = np.load(results_dir+m+'_ids.npy')
        q = np.argsort(ids)
        y = np.load(results_dir+m+'_y.npy')
        avg_preds = np.load(results_dir+m+'_holdout.npy')
        preds = np.array([avg_preds[i] for i in q])
        ytrue = np.array([y[i] for i in q])
        avg_tests = np.load(results_dir+m+'_test.npy')
        return ytrue, preds, avg_tests

    holdouts = []
    tests = []
    ys = []
    for m in models:
        y,h,t = load_model(m)
        if len(ys) == 0:
            ys.append(y)
        holdouts.append(h)
        tests.append(t)

    valid.loc[:,'Id'] = valid.Id.apply(lambda x : x.split('_clas')[0])


    val_set = valid[['Id']].merge(val_meta, on='Id', how='left').sort_values('Id').reset_index(drop=True)
    for mod_ix in range(len(models)):
        for i in range(28):
            val_set[models[mod_ix]+'_'+str(i)] = holdouts[mod_ix][:,i]


    # In[254]:


    test_set = test[['Id']].merge(test_meta, on='Id', how='left').reset_index(drop=True)
    for mod_ix in range(len(models)):
        for i in range(28):
            test_set[models[mod_ix]+'_'+str(i)] = tests[mod_ix][:,i]

    train_df = pd.read_csv(data_dir + 'train.csv')

    qqdata = train_df.copy()
    labels = np.zeros((qqdata.shape[0], 28), dtype=np.int32)
    if "Target" in qqdata:
        for i, lbls in qqdata['Target'].str.split().iteritems():
            for j in map(int, lbls):
                labels[i, j] = 1
    others_mult = []
    for j in range(28):
        others_mult.append((labels.shape[0]-labels[:,j].sum())/labels[:,j].sum())
        print(j,'\t',name_label_dict[j], '\t', labels[:,j].sum(), '\t', labels.shape[0]/labels[:,j].sum())
    others_mult


    from sklearn.metrics import roc_auc_score

    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.metrics import roc_auc_score

    def fold_in_fold(train_set, y, val_set, test_set, num_folds=5, num_iters=5, class_weight = None):
        test_predict = np.zeros(test_set.shape[0])
        val_predict = np.zeros(val_set.shape[0])
        all_scores = []
        for kk in range(num_iters):
            kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=239 + kk*101)
            for ho_trn_inx, ho_val_inx in kf.split(y,y):
                evals_result = {}
                param = {
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'scale_pos_weight' : class_weight,
                    'metric' : 'auc',
                    'learning_rate': 0.01, # 0.1
                    'colsample': 0.6,
                    'num_leaves': 256,
                    'min_data_in_leaf': 15,
                    'max_depth': 9,
                    'max_bin': 255, # 255
                    'bagging_fraction': 0.9,
                    'bagging_freq': 5, # 0
                    #'lambda_l1': 0.001,
                    #'lambda_l2': 0.005
                }

                trn_data = lgb.Dataset(train_set[ho_trn_inx,:], label=y[ho_trn_inx])
                val_data = lgb.Dataset(train_set[ho_val_inx,:], label=y[ho_val_inx])

                clf = lgb.train(param, trn_data, num_boost_round=1000, early_stopping_rounds=200,
                            valid_sets=[trn_data,val_data], valid_names=["train","val"], #feval = lgb_f1_score,
                            evals_result=evals_result, verbose_eval=None)
                fold_pred = clf.predict(train_set[ho_val_inx,:], num_iteration=clf.best_iteration)

                t_min = np.min(fold_pred)
                t_max = np.max(fold_pred)
                thresholds = np.linspace(t_min, t_max, 1000)
                scores = np.array([
                    f1_score(y[ho_val_inx], np.int32(fold_pred >= threshold), average='macro') for threshold in thresholds
                    if (np.sum(np.int32(fold_pred >= threshold)) > 0)
                ])

                threshold_best_index = np.argmax(scores)
                all_scores.append(scores[np.argmax(scores)])
                test_predict += (clf.predict(test_set, num_iteration=clf.best_iteration) >= thresholds[threshold_best_index]).astype(float)/(num_folds*num_iters)
                val_predict += (clf.predict(val_set, num_iteration=clf.best_iteration) >= thresholds[threshold_best_index]).astype(float)/(num_folds*num_iters)
        #print(np.mean(all_scores))
        return val_predict, test_predict

    from sklearn import metrics
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report

    y = ys[0].copy()
    num_folds = 5
    test_lgb_prob2 = np.zeros((test_set.values.shape[0],28))
    classes_thresholds = []
    classes_scores = []
    KK_ITER = 3
    class_thresholds = np.zeros(28)

    np.random.seed(239)

    for i in range(28):
        print()
        for kk in range(KK_ITER):
            pos_class = np.sum(y[:,i])
            neg_class = int(pos_class*others_mult[i])
            neg_idxs = np.array(range(len(y)))[np.where(y[:,i]==0,True,False).tolist()]
            np.random.shuffle(neg_idxs)
            idxs = np.hstack([neg_idxs[:neg_class],np.array(range(len(y)))[np.where(y[:,i]==1,True,False).tolist()]])
            np.random.shuffle(idxs)
            y_true = y[idxs]
            class_val_set = val_set.loc[idxs,:].reset_index(drop=True)
            class_weight = neg_class / (pos_class + neg_class)
            kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=239 + kk * 57)
            fold_inx = 1
            tst_lgb_prob_fold = np.zeros(test_lgb_prob2.shape)
            lgb_prob = np.zeros((class_val_set.values.shape[0], 28))
            for ho_trn_inx, ho_val_inx in kf.split(y_true,y_true[:,i]):
                val_pred,test_pred, = fold_in_fold(class_val_set.values[ho_trn_inx,1:],
                                                      y_true[ho_trn_inx,i],
                                                      class_val_set.values[ho_val_inx,1:],
                                                      test_set.values[:,1:],
                                                      3,5,class_weight
                                                     )
                lgb_prob[ho_val_inx,i] = val_pred
                tst_lgb_prob_fold[:,i] += test_pred/num_folds
                fold_inx += 1
            thresholds = np.linspace(0, 1, 1000)
            scores = np.array([
                f1_score(y_true[:,i], np.int32(lgb_prob[:,i] >= threshold), average='macro') for threshold in thresholds \
                if np.sum(np.int32(lgb_prob[:,i] >= threshold)) > 0
            ])
            threshold_best_index = np.argmax(scores)
            cl_thr = thresholds[threshold_best_index]
            #print(classification_report(y_true[:,i], np.int32(lgb_prob[:,i] >= cl_thr).astype(int)))
            class_thresholds[i] += cl_thr / (1.0 * KK_ITER)
            test_lgb_prob2[:,i] += tst_lgb_prob_fold[:,i] / (1.0*KK_ITER)
            print("Class {0} F1 score (0.5 threshold): {1:.5f}".format(name_label_dict[i], f1_score(y_true[:,i],(lgb_prob[:,i]>=0.5).astype(int), average='macro')))
            print("Class {0} F1 score ({1:.3f} threshold): {2:.5f}".format(name_label_dict[i], cl_thr, f1_score(y_true[:,i],(lgb_prob[:,i]>=cl_thr).astype(int), average='macro')))

    #patch 8 10

    iter_num = []
    num_outter_folds = []
    num_inner_folds = []
    for i in range(28):
        q = ys[0][:,i].sum()
        t = 3
        out_folds = 7
        in_folds = 7
        if q < 15:
            t = 20
            out_folds = 5
            in_folds = 3
        elif q < 110:
            t = 10
            out_folds = 5
            in_folds = 5
        elif q < 250:
            t = 7
            out_folds = 7
            in_folds = 5
        elif q < 1000:
            t = 5
        num_outter_folds.append(out_folds)
        num_inner_folds.append(in_folds)
        iter_num.append(t)
        print(i,q,t)

    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    feature_importances = {}
    for i in range(28):
        feature_importances[i] = []

    from sklearn.metrics import roc_curve, precision_recall_curve
    def threshold_search(y_true, y_proba, plot=False):
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        thresholds = np.append(thresholds, 1.001)
        F = 2 / (1/precision + 1/recall)
        best_score = np.max(F)
        best_th = thresholds[np.argmax(F)]
        if plot:
            plt.plot(thresholds, F, '-b')
            plt.plot([best_th], [best_score], '*r')
            plt.show()
        search_result = {'t': best_th , 'f1': best_score}
        return search_result

    def fold_in_fold(train_set, y, val_set, test_set, num_folds=5, num_iters=5,
                     class_weight = None, class_id = None, use_ridge = True):
        global feature_importances
        test_predict = np.zeros(test_set.shape[0])
        val_predict = np.zeros(val_set.shape[0])
        all_scores = []
        for kk in range(num_iters):
            kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=239 + kk*101)
            ridge_train = np.zeros((train_set.shape[0],1))
            ridge_val = np.zeros((val_set.shape[0],1))
            ridge_test = np.zeros((test_set.shape[0],1))
            #print()
            if use_ridge:
                for ho_trn_inx, ho_val_inx in kf.split(y,y):
                    scaler = StandardScaler()
                    scaler.fit(train_set[ho_trn_inx,:])
                    model = Ridge(alpha=0.1)
                    model.fit(scaler.transform(train_set[ho_trn_inx,:]), y[ho_trn_inx])
                    ridge_train[ho_val_inx,:] = model.predict(scaler.transform(train_set[ho_val_inx,:])).reshape(-1,1)
                    res = threshold_search(y[ho_val_inx], ridge_train[ho_val_inx,:])

                    #print('f1', res['f1'])

                    # Предсказываем голосованием тест, сразу усредняя по внутреннему фолду и внутренней итерации.
                    test_predict += ((model.predict(scaler.transform(test_set))
                                     >= res['t']).astype(float)/(num_folds*num_iters))

                    # Предсказываем голосованием валидацию, сразу усредняя по внутреннему фолду и внутренней итерации.
                    val_predict += ((model.predict(scaler.transform(val_set))
                                     >= res['t']).astype(float)/(num_folds*num_iters))
            else:
                for ho_trn_inx, ho_val_inx in kf.split(y,y):
                    evals_result = {}
                    param = {
                        'boosting_type': 'gbdt',
                        'objective': 'binary',
                        #'scale_pos_weight' : class_weight,
                        'metric' : 'auc', # Я хз, почему именно AUC.
                        'learning_rate': 0.001,
                        'colsample': 0.6,
                        'num_leaves': 12,
                        'min_data_in_leaf': 150,
                        'max_depth': 4,
                        'l1':1,'l2':10,
                        'max_bin': 1000,
                        'bagging_fraction': 0.9,
                        'bagging_freq': 5,
                        'min_sum_hessian_in_leaf ':1000,
                    }

                    # Формируем LGB-шные объекты датасетов для трейна и валидации.
                    trn_data = lgb.Dataset(train_set[ho_trn_inx,:], label=y[ho_trn_inx])
                    val_data = lgb.Dataset(train_set[ho_val_inx,:], label=y[ho_val_inx])

                    # Учимся с ES на 1000 деревьев. Возможно, есть смысл поставить 1500 на всякий случай?
                    clf = lgb.train(param, trn_data, num_boost_round=1000, early_stopping_rounds=200,
                                valid_sets=[trn_data,val_data], valid_names=["train","val"],
                                evals_result=evals_result, verbose_eval=100)

                    # Предсказываем валидационный фолд.
                    fold_pred = clf.predict(train_set[ho_val_inx,:], num_iteration=clf.best_iteration)

                    # Перебираем пороги для f1.
                    t_min = np.min(fold_pred)
                    t_max = np.max(fold_pred)
                    thresholds = np.linspace(t_min, t_max, 1000)
                    scores = np.array([
                        f1_score(y[ho_val_inx], np.int32(fold_pred >= threshold))
                        for threshold in thresholds
                        if (np.sum(np.int32(fold_pred >= threshold)) > 0)
                    ])

                    # Выбираем лучший порог.
                    threshold_best_index = np.argmax(scores)
                    print('f1', scores[np.argmax(scores)])

                    # Предсказываем голосованием тест, сразу усредняя по внутреннему фолду и внутренней итерации.
                    test_predict += ((clf.predict(test_set, num_iteration=clf.best_iteration)
                                     >= thresholds[threshold_best_index]).astype(float)/(num_folds*num_iters))

                    # Предсказываем голосованием валидацию, сразу усредняя по внутреннему фолду и внутренней итерации.
                    val_predict += ((clf.predict(val_set, num_iteration=clf.best_iteration)
                                     >= thresholds[threshold_best_index]).astype(float)/(num_folds*num_iters))
        #print(np.mean(all_scores))
        return val_predict, test_predict

    # from sklearn import metrics
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report

    y = ys[0].copy()
    num_folds = 5
    test_lgb_prob2 = np.zeros((test_set.values.shape[0],28))
    classes_thresholds = []
    classes_scores = []

    class_thresholds = np.zeros(28)
    class_scores = np.zeros(28)

    np.random.seed(239)

    for i in [8,10]:
        print()
        KK_ITER = iter_num[i]
        num_folds = num_outter_folds[i]
        for kk in range(KK_ITER):
            pos_class = np.sum(y[:,i])
            neg_class = int(pos_class*others_mult[i])
            neg_idxs = np.array(range(len(y)))[np.where(y[:,i]==0,True,False).tolist()]
            np.random.shuffle(neg_idxs)
            idxs = np.hstack([neg_idxs[:neg_class],np.array(range(len(y)))[np.where(y[:,i]==1,True,False).tolist()]])
            np.random.shuffle(idxs)
            y_true = y[idxs]
            class_val_set = val_set.loc[idxs,:].reset_index(drop=True)
            class_weight = neg_class / pos_class
            kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=239 + kk * 57)
            tst_lgb_prob_fold = np.zeros(test_lgb_prob2.shape)
            lgb_prob = np.zeros((class_val_set.values.shape[0], 28))
            own_set = list(range(class_val_set.shape[1]))[1:]
            if np.sum(y[:,i]) < 15:
                own_set = list(set(list(range(1,15,1)) + [16+i+q*28 for q in range(len(models)-1)] + \
                                   [16 + 28*3 + q for q in range(28)] + [16+11*28+q for q in range(10)] + [16+11*28+10+q for q in range(39)]))
                #own_set = list(set(list(range(1,15,1)) + [16+i+q*28 for q in range(len(models)-1)]))
            for ho_trn_inx, ho_val_inx in kf.split(y_true,y_true[:,i]):
                val_pred,test_pred, = fold_in_fold(class_val_set.values[ho_trn_inx,:][:,own_set],
                                                      y_true[ho_trn_inx,i],
                                                      class_val_set.values[ho_val_inx,:][:,own_set],
                                                      test_set.values[:,:][:,own_set],
                                                      num_inner_folds[i],iter_num[i],class_weight,i
                                                     )
                lgb_prob[ho_val_inx,i] = val_pred
                tst_lgb_prob_fold[:,i] += test_pred/num_folds
            # Перебираем пороги для f1
            thresholds = np.linspace(0, 1, 1000)
            scores = np.array([
                f1_score(y_true[:,i], np.int32(lgb_prob[:,i] >= threshold), average='binary') # binary f1
                for threshold in thresholds
                if np.sum(np.int32(lgb_prob[:,i] >= threshold)) > 0 # Чтобы ворнинги не писались?
            ])

            # Выбираем трешхолд с лучшим f1.
            threshold_best_index = np.argmax(scores)
            cl_thr = thresholds[threshold_best_index]

            # Трешхолды усредняем по итерациям.
            class_thresholds[i] += cl_thr / (1.0 * KK_ITER)

            # Предикты тоже усредняем по итерациям.
            test_lgb_prob2[:,i] += tst_lgb_prob_fold[:,i] / (1.0*KK_ITER)

            # Печатаем скор валидации для порога, равного 0.5.
            print("Class {0} F1 score (0.5 threshold): {1:.5f}"
                  .format(
                      name_label_dict[i],
                      f1_score(y_true[:,i],(lgb_prob[:,i]>=0.5).astype(int), average='binary')))

            # Печатаем скор валидации для лучшего порога по этому фолду.
            print("Class {0} F1 score ({1:.3f} threshold): {2:.5f}"
                  .format(
                      name_label_dict[i],
                      cl_thr,
                      f1_score(y_true[:,i],(lgb_prob[:,i]>=cl_thr).astype(int), average='binary')
                  ))

    #patch 27
    models = ['sf_fastai_inf_seresnext','sf_fastai_inf_incv4','pd_seresnext_v5','sf_fastai_inf_sev2',
              'db_seresnext_v3_LSEP4','pd_seresnext_v6','pd_inceptionv4_v2',
              'pd_seresnext_v7','pd_xception_v1', 'pd_seresnext_v10','pd_bninception_v3','pd_seresnext_ovr_v3']
    results_dir = 'results/'

    holdouts = []
    tests = []
    ys = []
    for m in models:
        y,h,t = load_model(m)
        if len(ys) == 0:
            ys.append(y)
        holdouts.append(h)
        tests.append(t)
    valid.loc[:,'Id'] = valid.Id.apply(lambda x : x.split('_clas')[0])
    val_set = valid[['Id']].merge(val_meta, on='Id', how='left').sort_values('Id').reset_index(drop=True)
    for mod_ix in range(len(models)):
        for i in range(holdouts[mod_ix].shape[1]):
            val_set[models[mod_ix]+'_'+str(i)] = holdouts[mod_ix][:,i]
    test_set = test[['Id']].merge(test_meta, on='Id', how='left').reset_index(drop=True)
    for mod_ix in range(len(models)):
        for i in range(tests[mod_ix].shape[1]):
            test_set[models[mod_ix]+'_'+str(i)] = tests[mod_ix][:,i]

    def fold_in_fold(train_set, y, val_set, test_set, num_folds=5, num_iters=5,
                     class_weight = None, class_id = None, use_ridge = True):
        global feature_importances
        test_predict = np.zeros(test_set.shape[0])
        val_predict = np.zeros(val_set.shape[0])
        all_scores = []
        for kk in range(num_iters):
            kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=239 + kk*101)
            ridge_train = np.zeros((train_set.shape[0],1))
            ridge_val = np.zeros((val_set.shape[0],1))
            ridge_test = np.zeros((test_set.shape[0],1))
            #print()
            if use_ridge:
                for ho_trn_inx, ho_val_inx in kf.split(y,y):
                    scaler = StandardScaler()
                    scaler.fit(train_set[ho_trn_inx,:])
                    model = Ridge(alpha=.1) #, solver='svd'
                    model.fit(scaler.transform(train_set[ho_trn_inx,:]), y[ho_trn_inx])
                    ridge_train[ho_val_inx,:] = model.predict(scaler.transform(train_set[ho_val_inx,:])).reshape(-1,1)
                    res = threshold_search(y[ho_val_inx], ridge_train[ho_val_inx,:])

                    #print('f1', res['f1'])

                    # Предсказываем голосованием тест, сразу усредняя по внутреннему фолду и внутренней итерации.
                    test_predict += ((model.predict(scaler.transform(test_set))
                                     >= res['t']*1.05).astype(float)/(num_folds*num_iters))

                    # Предсказываем голосованием валидацию, сразу усредняя по внутреннему фолду и внутренней итерации.
                    val_predict += ((model.predict(scaler.transform(val_set))
                                     >= res['t']*1.05).astype(float)/(num_folds*num_iters))
            else:
                for ho_trn_inx, ho_val_inx in kf.split(y,y):
                    evals_result = {}
                    param = {
                        'boosting_type': 'gbdt',
                        'objective': 'binary',
                        #'scale_pos_weight' : class_weight,
                        'metric' : 'auc', # Я хз, почему именно AUC.
                        'learning_rate': 0.001,
                        'colsample': 0.6,
                        'num_leaves': 12,
                        'min_data_in_leaf': 150,
                        'max_depth': 4,
                        'l1':1,'l2':10,
                        'max_bin': 1000,
                        'bagging_fraction': 0.9,
                        'bagging_freq': 5,
                        'min_sum_hessian_in_leaf ':1000,
                    }

                    # Формируем LGB-шные объекты датасетов для трейна и валидации.
                    trn_data = lgb.Dataset(train_set[ho_trn_inx,:], label=y[ho_trn_inx])
                    val_data = lgb.Dataset(train_set[ho_val_inx,:], label=y[ho_val_inx])

                    # Учимся с ES на 1000 деревьев. Возможно, есть смысл поставить 1500 на всякий случай?
                    clf = lgb.train(param, trn_data, num_boost_round=1000, early_stopping_rounds=200,
                                valid_sets=[trn_data,val_data], valid_names=["train","val"],
                                evals_result=evals_result, verbose_eval=100)

                    # Предсказываем валидационный фолд.
                    fold_pred = clf.predict(train_set[ho_val_inx,:], num_iteration=clf.best_iteration)

                    # Перебираем пороги для f1.
                    t_min = np.min(fold_pred)
                    t_max = np.max(fold_pred)
                    thresholds = np.linspace(t_min, t_max, 1000)
                    scores = np.array([
                        f1_score(y[ho_val_inx], np.int32(fold_pred >= threshold))
                        for threshold in thresholds
                        if (np.sum(np.int32(fold_pred >= threshold)) > 0)
                    ])

                    # Выбираем лучший порог.
                    threshold_best_index = np.argmax(scores)
                    print('f1', scores[np.argmax(scores)])

                    # Предсказываем голосованием тест, сразу усредняя по внутреннему фолду и внутренней итерации.
                    test_predict += ((clf.predict(test_set, num_iteration=clf.best_iteration)
                                     >= thresholds[threshold_best_index]).astype(float)/(num_folds*num_iters))

                    # Предсказываем голосованием валидацию, сразу усредняя по внутреннему фолду и внутренней итерации.
                    val_predict += ((clf.predict(val_set, num_iteration=clf.best_iteration)
                                     >= thresholds[threshold_best_index]).astype(float)/(num_folds*num_iters))
        #print(np.mean(all_scores))
        return val_predict, test_predict

    iter_num[27] = 40

    for i in [27]:
        print()
        KK_ITER = iter_num[i]
        num_folds = num_outter_folds[i]
        for kk in range(KK_ITER):
            pos_class = np.sum(y[:,i])
            neg_class = int(pos_class*others_mult[i])
            neg_idxs = np.array(range(len(y)))[np.where(y[:,i]==0,True,False).tolist()]
            np.random.shuffle(neg_idxs)
            idxs = np.hstack([neg_idxs[:neg_class],np.array(range(len(y)))[np.where(y[:,i]==1,True,False).tolist()]])
            np.random.shuffle(idxs)
            y_true = y[idxs]
            class_val_set = val_set.loc[idxs,:].reset_index(drop=True)
            class_weight = neg_class / pos_class
            kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=239 + kk * 57)
            tst_lgb_prob_fold = np.zeros(test_lgb_prob2.shape)
            lgb_prob = np.zeros((class_val_set.values.shape[0], 28))
            own_set = list(range(class_val_set.shape[1]))[1:]
            if np.sum(y[:,i]) < 15:
                #own_set = list(set(list(range(1,15,1)) + [16+i+q*28 for q in range(len(models)-1)] + \
                #                   [16 + 28*3 + q for q in range(28)] + [16+11*28+q for q in range(10)] + [16+11*28+10+q for q in range(39)]))
                #own_set = list(set(list(range(1,15,1)) + [16+i+q*28 for q in range(len(models)-1)]))
                own_set = list(set([16+i+q*28 for q in range(len(models)-1)]))
            for ho_trn_inx, ho_val_inx in kf.split(y_true,y_true[:,i]):
                val_pred,test_pred, = fold_in_fold(class_val_set.values[ho_trn_inx,:][:,own_set],
                                                      y_true[ho_trn_inx,i],
                                                      class_val_set.values[ho_val_inx,:][:,own_set],
                                                      test_set.values[:,:][:,own_set],
                                                      num_inner_folds[i],iter_num[i],class_weight,i
                                                     )
                lgb_prob[ho_val_inx,i] = val_pred
                tst_lgb_prob_fold[:,i] += test_pred/num_folds
            # Перебираем пороги для f1
            thresholds = np.linspace(0, 1, 1000)
            scores = np.array([
                f1_score(y_true[:,i], np.int32(lgb_prob[:,i] >= threshold), average='binary') # binary f1
                for threshold in thresholds
                if np.sum(np.int32(lgb_prob[:,i] >= threshold)) > 0 # Чтобы ворнинги не писались?
            ])

            # Выбираем трешхолд с лучшим f1.
            threshold_best_index = np.argmax(scores)
            cl_thr = thresholds[threshold_best_index]

            # Трешхолды усредняем по итерациям.
            class_thresholds[i] += cl_thr / (1.0 * KK_ITER)

            # Предикты тоже усредняем по итерациям.
            test_lgb_prob2[:,i] += tst_lgb_prob_fold[:,i] / (1.0*KK_ITER)

            # Печатаем скор валидации для порога, равного 0.5.
            print("Class {0} F1 score (0.5 threshold): {1:.5f}"
                  .format(
                      name_label_dict[i],
                      f1_score(y_true[:,i],(lgb_prob[:,i]>=0.5).astype(int), average='binary')))

            # Печатаем скор валидации для лучшего порога по этому фолду.
            print("Class {0} F1 score ({1:.3f} threshold): {2:.5f}"
                  .format(
                      name_label_dict[i],
                      cl_thr,
                      f1_score(y_true[:,i],(lgb_prob[:,i]>=cl_thr).astype(int), average='binary')
                  ))

    #submit
    ppreds = test_pred.copy()
    for i in range(28):
        ppreds[:,i] = ppreds[:,i] >= class_thresholds[i]

    mdict = {}
    for i,p in zip(ids,ppreds):
        mdict[i] = ' '.join([str(q) for q in np.argwhere(p).ravel()])
    sub = pd.DataFrame.from_dict(mdict,orient='index').reset_index()
    sub.columns = ['Id','Predicted']

    japanese_duplicates = pd.read_csv('input/leak_v4.csv')
    japanese_duplicates.columns = ['Test','C','H','Target']

    submit = pd.read_csv(data_dir + "sample_submission.csv").drop(['Predicted'], axis=1)
    submit = submit.merge(sub, on='Id',how='left')

    print(len(submit.loc[submit.Predicted == '',:]))

    for i, row in japanese_duplicates.iterrows():
        test_dup_id = row['Test']
        test_dup_classes = row['Target']
        submit.loc[submit['Id'] == test_dup_id, 'Predicted'] = test_dup_classes

    submit.to_csv('submits/' + 'submission.csv',index=False)
    submit.head(n=30)
