from utils.include import *
from utils.metric import do_kaggle_metric
from utils.data import ProteinDataset
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from nn_models.aggregated_classifier import AggregatedClassifier as Net
import sys
#INPUT_DIR = '/home/t-zhga/protein-kaggle/input'
#out_dir = '../model_weights/res18_256'
#initial_checkpoint = out_dir + '/checkpoint/epoch_142_loss_0.4630_cv_0.6798_model.pth'
INPUT_DIR = sys.argv[1]
out_dir = sys.argv[2]
initial_checkpoint = out_dir  + sys.argv[3]


valid_augment = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(lambda image: [transforms.RandomHorizontalFlip()(image),transforms.RandomVerticalFlip()(image)]),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.08069, 0.05258, 0.05487, 0.08282], std=[0.13704, 0.10145, 0.15313, 0.13814])(transforms.ToTensor()(crop)) for crop in crops])),
    ])
test_augment = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(lambda image: [transforms.RandomHorizontalFlip()(image),transforms.RandomVerticalFlip()(image)]),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.08069, 0.05258, 0.05487, 0.08282], std=[0.13704, 0.10145, 0.15313, 0.13814])(transforms.ToTensor()(crop)) for crop in crops])),
    ])

def run_predict():
    os.makedirs(out_dir +'/test' , exist_ok=True)
    csv_file = out_dir + '/test/%s-%s-tencrop-0.2.csv'%(out_dir.split('/')[-1], initial_checkpoint.split('/')[-1])
    batch_size = 16

    net = Net().cuda()

    if initial_checkpoint is not None:
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    net.set_mode('valid')

    ##--------------------------------------------------------
    # print('****Choose appropriate prediction threshold maximalizing the validation F1-score****')
    # df = pd.read_csv(os.path.join(INPUT_DIR, 'train_merge.csv'), engine='python')
    # df['Target'] = [[int(i) for i in s.split()] for s in df['Target']]
    # df['target_vec_float'] = df['Target'].apply(lambda label: np.eye(len(label_map_dict),dtype=np.float)[label].sum(axis=0))
    # # print(df.head())
    # msss = MultilabelStratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    # X, y = df['Id'].tolist(), df['target_vec_float'].tolist()

    # for split, (train_index, val_index) in enumerate(msss.split(X,y)):
    #     print('**********Split %d**********'%split)
    #     print('train_num: ', len(train_index), 'valid_num: ', len(val_index))
    #     train_df = df.loc[df.index.intersection(train_index)].copy()
    #     val_df = df.loc[df.index.intersection(val_index)].copy()
    #     gc.collect()
    #     print('********Valid set forward********')
    #     valid_dataset = ProteinDataset(val_df, valid_augment, 'valid')
    #     valid_loader  = DataLoader(
    #                             valid_dataset,
    #                             sampler     = RandomSampler(valid_dataset),
    #                             batch_size  = batch_size,
    #                             drop_last   = False,
    #                             num_workers = 8,
    #                             pin_memory  = True)
    #     predicts = []
    #     truths   = []
    #     for input, truth, image_id in tqdm(valid_loader):
    #         bs, ncrops, c, h, w = input.size()
    #         input = input.cuda()
    #         truth = truth.cuda()

    #         with torch.no_grad():
    #             logit = data_parallel(net, input.view(-1, c, h, w))
    #             logit_avg = logit.view(bs, ncrops, -1).mean(1)
    #             prob  = F.sigmoid(logit_avg)

    #         predicts.append(prob.data.cpu().numpy())
    #         truths.append(truth.data.cpu().numpy())

    #     predicts = np.concatenate(predicts).squeeze()
    #     truths   = np.concatenate(truths).squeeze()
    #     print(predicts.shape, truths.shape)
    #     print('********Search thredsholds********')
    #     best_cv, best_threshold  = do_kaggle_metric(predicts, truths)
    #     print('Probability threshold maximizing CV F1-score for each class:')
    #     print(best_threshold)
    #     print('best_cv:', best_cv)


    sample_sub_df = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'), engine='python')
    test_dataset = ProteinDataset(sample_sub_df, test_augment, 'test')
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)
    assert(len(test_dataset)>=batch_size)

    all_prob = []
    all_id = []
    for input, truth, image_id in tqdm(test_loader):
        bs, ncrops, c, h, w = input.size()
        input = input.cuda()
        truth = truth.cuda()
        with torch.no_grad():
            logit = data_parallel(net, input.view(-1, c, h, w))
            logit_avg = logit.view(bs, ncrops, -1).mean(1)
            prob  = F.sigmoid(logit_avg)
        prob = prob.squeeze().data.cpu().numpy()
        all_prob.append(prob)
        all_id.append(image_id)
    all_prob = np.concatenate(all_prob)
    all_id = np.concatenate(all_id).tolist()
    assert(all_id == sample_sub_df['Id'].tolist())
    all_pred = []
    thredshold = 0.2#best_threshold
    for prob in all_prob:
        s = ' '.join(list([str(i) for i in np.nonzero(prob>thredshold)[0]]))
        all_pred.append(s)
    sub_df = pd.DataFrame({ 'Id' : all_id , 'Predicted' : all_pred}).astype(str)
    sub_df.to_csv(csv_file, header=True, index=False)
    # all_prob = (all_prob*255).astype(np.uint8)
    np.save( out_dir + '/test/%s-%s-prob-tencrop.npy'%(out_dir.split('/')[-1], initial_checkpoint.split('/')[-1]),all_prob)
    print('Succcess!')

if __name__ == '__main__':
    run_predict()
