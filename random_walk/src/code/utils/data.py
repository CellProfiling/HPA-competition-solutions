from utils.include import *

def open_rgby(path,id): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id+'_'+color+'.png'), flags).astype(np.uint8)#np.float32
           for color in colors]
    return np.stack(img, axis=-1)

class ProteinDataset(Dataset):
    """docstring for ProteinDataset"""
    def __init__(self, split, augmentation, mode = 'train'):
        super(ProteinDataset, self).__init__()
        self.split = split
        self.aug = augmentation
        self.mode = mode
        # if self.mode in ['valid']:
        #     self.split['target_vec_float'] = self.split['Target'].apply(lambda label: np.eye(len(label_map_dict),dtype=np.float)[label].sum(axis=0))

    def __getitem__(self, index):
        image_id = self.split.iloc[index]['Id']
        if self.mode in ['train', 'valid']:
            image = open_rgby(INPUT_DIR+'/train', image_id)
            label = self.split.iloc[index]['target_vec_float']
        elif self.mode in ['test']:
            image = open_rgby(INPUT_DIR+'/test', image_id)
            label = 2
        image =self.aug(image)
        return image, label, image_id

    def __len__(self):
        return len(self.split)

# class ProteinDataset(Dataset):
#     """docstring for ProteinDataset"""
#     def __init__(self, split, augmentation, mode = 'train'):
#         super(ProteinDataset, self).__init__()
#         self.split = split
#         self.aug = augmentation
#         self.mode = mode
#         self.df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'), engine='python').set_index('Id')
#         self.df['Target'] = [[int(i) for i in s.split()] for s in self.df['Target']]
#         self.df['target_vec_float'] = self.df['Target'].apply(lambda label: np.eye(len(label_map_dict), dtype=np.float32)[label].sum(axis=0))
#         # if self.mode in ['valid']:
#         #     self.split['target_vec_float'] = self.split['Target'].apply(lambda label: np.eye(len(label_map_dict),dtype=np.float)[label].sum(axis=0))

#     def __getitem__(self, index):
#         image_id = self.split[index]
#         if self.mode in ['train', 'valid']:
#             image = open_rgby(INPUT_DIR+'/train', image_id)
#             label = self.df.loc[image_id]['target_vec_float']
#         elif self.mode in ['test']:
#             image = open_rgby(INPUT_DIR+'/test', image_id)
#             label = 2
#         image =self.aug(image)
#         return image, label, image_id

#     def __len__(self):
#         return len(self.split)

