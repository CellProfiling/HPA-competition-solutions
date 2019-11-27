import os
import sys

DATASET_PATH_512='./data/humanprotein/input/test/'
DATASET_PATH_1024='./data/humanprotein/input1024/test/'
# DATASET_PATH_512='/ai/local/lichuanpeng/DataSet/Kaggle/HumanProtein/input/test/'
# DATASET_PATH_1024='/ai/local/lichuanpeng/DataSet/Kaggle/HumanProtein/input1024/test/'

class BaseConfig():
    def __init__(self,base_dir,model_name):

        self.set_path(dataset_type='official')
        self.sample_submit_file = "./data/sample_submission.csv"
        self.weights = base_dir + "/models"
        self.best_models = base_dir + "/best_models/"
        self.submit = base_dir + "/submit/"
        self.logs_dir = base_dir + '/logs'
        self.features_dir = base_dir + '/features'
        self.submit_file = os.path.join(self.submit, '{}_submission.csv'.format(model_name))
        self.cropindex = -1
        self.run_type = 'train'
        mkdir_with_check(base_dir)
        mkdir_with_check(self.weights)
        mkdir_with_check(self.best_models)
        mkdir_with_check(self.submit)
        mkdir_with_check(self.logs_dir)
        mkdir_with_check(self.features_dir)

    def set_path(self,dataset_type='official',image_size=512):
        if image_size>512:
            self.test_data =DATASET_PATH_1024
        else:
            self.test_data = DATASET_PATH_512


    def save_param(self):
        with open(os.path.join(self.weights,'train_params.txt'),'wb') as fd:
            for name, value in vars(self).items():
                print('{}:{}'.format(name,value))
                fd.write('{}:{}\n'.format(name,value))

def mkdir_with_check(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        try:
            os.system('chmod 777 {}'.format(dir_path))
        except:
            pass

class DefaultConfigs_xception(BaseConfig):
    def __init__(self,model_name,gpu_id,tag='',lr=0.05,epoch=50):

        self.model_name = model_name
        self.BASE_DIR = "./models/{}/{}{}".format(self.model_name,self.model_name,'_'+tag if tag!='' else '')
        BaseConfig.__init__(self,self.BASE_DIR,self.model_name)

        self.num_classes = 28
        self.img_weight = 512
        self.img_height = 512
        self.channels = 4
        self.lr = lr
        self.batch_size = 14
        self.epochs = epoch
        self.cuda_id = gpu_id


class DefaultConfigs_inceptionv3(BaseConfig):
    def __init__(self,model_name,gpu_id,tag='',lr=0.05,epoch=50):

        self.model_name = model_name
        self.BASE_DIR = "./models/{}/{}{}".format(self.model_name,self.model_name,'_'+tag if tag!='' else '')
        BaseConfig.__init__(self,self.BASE_DIR,self.model_name)

        self.num_classes = 28
        self.img_weight = 512
        self.img_height = 512
        self.channels = 4
        self.lr = lr
        self.batch_size = 28
        self.epochs = epoch
        self.cuda_id = gpu_id


class DefaultConfigs_inceptionv4(BaseConfig):
    def __init__(self,model_name,gpu_id,tag='',lr=0.05,epoch=50):

        self.model_name = model_name
        self.BASE_DIR = "./models/{}/{}{}".format(self.model_name,self.model_name,'_'+tag if tag!='' else '')
        BaseConfig.__init__(self,self.BASE_DIR,self.model_name)

        self.num_classes = 28
        self.img_weight = 512
        self.img_height = 512
        self.channels = 4
        self.lr = lr
        self.batch_size = 20
        self.epochs = epoch
        self.cuda_id = gpu_id


def get_config(model_name,gpu_id,tag='',lr=0.05,epoch=50):
    if model_name == 'xception' or model_name.startswith('xception'):
        return DefaultConfigs_xception(model_name,gpu_id,tag,lr,epoch)
    elif model_name == 'inceptionv3' or model_name.startswith('inceptionv3'):
        return DefaultConfigs_inceptionv3(model_name,gpu_id,tag,lr,epoch)
    elif model_name == 'inceptionv4' or model_name.startswith('inceptionv4'):
        return DefaultConfigs_inceptionv4(model_name,gpu_id,tag,lr,epoch)
    else:
        print('Error config {} not found!'.format(model_name))
        sys.exit('0')

