from utils.include import *
from utils.rate import *
from utils.data import ProteinDataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from utils.metric import do_kaggle_metric

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

from nn_models.aggregated_classifier import AggregatedClassifier as Net

train_augment = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(256),
        transforms.RandomChoice([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()]),
        transforms.RandomCrop(256), # 256, 384, 512
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.08069, 0.05258, 0.05487, 0.08282], std=[0.13704, 0.10145, 0.15313, 0.13814]),
        # transforms.Normalize(mean=[0.0503, 0.0450, 0.0220, 0.0910], std=[0.0967, 0.0886, 0.0902, 0.1493]),
    ])

valid_augment = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.08069, 0.05258, 0.05487, 0.08282], std=[0.13704, 0.10145, 0.15313, 0.13814]),
])

def do_valid( net, valid_loader ):

    valid_num  = 0
    valid_loss = np.zeros(2,np.float32)

    predicts = []
    truths   = []
    probs = []

    for input, truth, image_id in valid_loader:
        # bs, ncrops, c, h, w = input.size()
        input = input.cuda()
        truth = truth.cuda()
        with torch.no_grad():
            # logit = data_parallel(net, input.view(-1, c, h, w))
            # logit_avg = logit.view(bs, ncrops, -1).mean(1)
            # prob  = torch.sigmoid(logit_avg)

            logit = data_parallel(net, input)
            prob  = torch.sigmoid(logit)

        batch_size = len(image_id)
        # valid_loss += batch_size*np.array(( loss.item(), 0))
        valid_num += batch_size

        predicts.append(logit.data.cpu().numpy())
        truths.append(truth.data.cpu().numpy())
        probs.append(prob.data.cpu().numpy())

    assert(valid_num == len(valid_loader.sampler))
    # valid_loss  = valid_loss/valid_num

    #--------------------------------------------------------
    predicts = np.concatenate(predicts).squeeze()
    truths   = np.concatenate(truths).squeeze()
    probs = np.concatenate(probs).squeeze()
    # print(predicts.shape, truths.shape)
    best_cv, best_threshold  = do_kaggle_metric(probs, truths)
    predicts = torch.from_numpy(predicts).cuda()
    truths = torch.from_numpy(truths).cuda()
    valid_loss[0] = net.compute_loss(predicts, truths)
    valid_loss[1] = best_cv
    return valid_loss


def run_train():
    out_dir = './results/resnet18/256'
    initial_checkpoint = None
    # initial_checkpoint = out_dir + '/checkpoint/epoch_98_loss_0.4410_cv_0.6916_model.pth'

    os.makedirs(out_dir +'/checkpoint', exist_ok=True)

    ## ** dataset setting **
    batch_size = 96
    batch_size_valid = 126#84
    df = pd.read_csv(os.path.join(INPUT_DIR, 'train_merge.csv'), engine='python')
    df['Target'] = [[int(i) for i in s.split()] for s in df['Target']]
    df['target_vec_float'] = df['Target'].apply(lambda label: np.eye(len(label_map_dict),dtype=np.float32)[label].sum(axis=0))
    # print(df.head())
    msss = MultilabelStratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    X, y = df['Id'].tolist(), df['target_vec_float'].tolist()

    for split, (train_index, val_index) in enumerate(msss.split(X,y)):
        print('**********Split %d**********'%split)
        print('train_num: ', len(train_index), 'valid_num: ', len(val_index))
        train_df = df.loc[df.index.intersection(train_index)].copy()
        val_df = df.loc[df.index.intersection(val_index)].copy()
        gc.collect()


        train_dataset = ProteinDataset(train_df, train_augment, 'train')
        train_loader  = DataLoader(
                            train_dataset,
                            sampler     = RandomSampler(train_dataset),
                            batch_size  = batch_size,
                            drop_last   = True,
                            num_workers = 8,
                            pin_memory  = True)

        valid_dataset = ProteinDataset(val_df, valid_augment, 'valid')
        valid_loader  = DataLoader(
                            valid_dataset,
                            sampler     = RandomSampler(valid_dataset),
                            batch_size  = batch_size_valid,
                            drop_last   = False,
                            num_workers = 8,
                            pin_memory  = True)

        assert(len(train_dataset)>=batch_size)

        ## ** net setting **
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # net.to(device)
        net = Net().cuda()
        print(net)


        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1, momentum=0.9,
                              weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cycle_epochs , eta_min=0.0005)

        if initial_checkpoint is not None:
            net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
            print('load model from', initial_checkpoint)

        start = timer()
        epochs = 150
        for epoch in range(epochs):
            # scheduler.step()
            print('Epoch:{}'.format(epoch))
            net.set_mode('train')
            sum_train_loss, num_batch = np.zeros(1,np.float32), 0
            for input, truth, image_id in tqdm(train_loader):
                # print(image_id)
                input = input.cuda()
                truth = truth.cuda()

                logit = data_parallel(net, input)
                loss  = net.compute_loss(logit, truth)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss = np.array((loss.item()))
                sum_train_loss += batch_loss
                num_batch += 1

            train_loss = sum_train_loss / num_batch
            rate = get_learning_rate(optimizer)
            net.set_mode('valid')
            valid_loss, valid_cv = do_valid(net, valid_loader)
            scheduler.step(valid_loss)
            print('\r%02d  %0.4f  |  %0.4f (%0.4f) |  %0.4f  |  %s ' % (\
                    epoch, rate,
                    valid_loss, valid_cv,
                    train_loss[0],
                    str(timer() - start)), end = '')
            # if rate<0.001:
            torch.save(net.state_dict(),out_dir +'/checkpoint/epoch_%02d_loss_%0.4f_cv_%0.4f_model.pth'%(epoch, valid_loss, valid_cv))
            print('\n save model to /checkpoint/epoch_%02d_loss_%0.4f_cv_%0.4f_model.pth'%(epoch, valid_loss, valid_cv))
        print('Success!')


if __name__ == '__main__':
    run_train()
