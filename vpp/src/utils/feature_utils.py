#coding=utf-8
import h5py
import os
import fcntl


class FeatureDataSet(object):

    def __init__(self,save_path,feature_dim,readonly=False):
        self.feature_dim = feature_dim
        self.save_path = save_path
        if not os.path.exists(save_path):
            h5f = h5py.File(save_path, 'w')
            h5f.create_dataset('data',shape=[0,feature_dim],maxshape=(None,feature_dim),dtype='float64')
            h5f.close()
        if readonly:
            self.h5f = h5py.File(save_path, 'r')
        else:
            self.h5f = h5py.File(save_path, 'a')
        self.dataset = self.h5f['data']
        self.feature_dim = self.dataset.shape[1]

    def num_features(self):
        return self.dataset.shape[0]

    def feature_shape(self):
        return self.dataset.shape

    def writeFeature(self,features):
        self.dataset.resize([self.dataset.shape[0]+features.shape[0],self.feature_dim])
        self.dataset[self.dataset.shape[0]-features.shape[0]:] = features

    def readFeature(self,start,end):
        return self.dataset[start:end]

    def read_all_feature(self):
        return self.dataset[:self.num_features()]

    def close(self):
        self.h5f.close()



def write_feature_2_h5(h5_file_path,sub_features):
    if os.path.exists(h5_file_path):
        dataset = FeatureDataSet(h5_file_path, feature_dim=0)
        assert dataset.feature_shape()[1]==sub_features.shape[1],'Error dim of h5 ({}) not equal to input feature ({})'.format(dataset.feature_shape()[1],sub_features.shape[1])
    else:
        dataset = FeatureDataSet(h5_file_path, feature_dim=sub_features.shape[1])

    start = dataset.num_features()
    dataset.writeFeature(sub_features)
    end = dataset.num_features()-1
    dataset.close()
    return start,end


def read_all_features(h5_file_path):
    assert os.path.exists(h5_file_path),'Error h5 file not found:'+h5_file_path
    dataset = FeatureDataSet(h5_file_path, feature_dim=0)
    features = dataset.readFeature(0,dataset.num_features())
    dataset.close()
    return features


def get_feature_shape(h5_file_path):
    assert os.path.exists(h5_file_path),'Error h5 file not found:'+h5_file_path
    dataset = FeatureDataSet(h5_file_path, feature_dim=0,readonly=True)
    feature_shape = dataset.feature_shape()
    dataset.close()
    return feature_shape

