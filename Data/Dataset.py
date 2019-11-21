# -*- coding: utf-8 -*-
# @Author  : LG

from torch.utils.data import Dataset
import h5py
import numpy as np

__all__ = ['indoor3d_Dataset']

class indoor3d_Dataset(Dataset):
    def __init__(self, is_train = True, data_root='Data', test_area=5):
        self.data_root = data_root

        if self.exists_data():
            train_data, train_label, test_data, test_label = self.recognize_all_data(test_area=test_area)
            if is_train:
                self.data = train_data
                self.label = train_label
            else:
                self.data = test_data
                self.label = test_label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def exists_data(self):
        import os
        import zipfile

        data_dir = '{}/indoor3d_sem_seg_hdf5_data'.format(self.data_root)
        if os.path.exists(data_dir):
            return True
        else:
            zip_file = '{}/indoor3d_sem_seg_hdf5_data.zip'.format(self.data_root)
            if os.path.exists(zip_file):
                print('Will unzip data to Data/')
                with zipfile.ZipFile(zip_file, 'r') as f:
                    for file in f.namelist():
                        print(file)
                        f.extract(file,'{}'.format(self.data_root))
                return True
            else:
                print('Please download data from https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip, '
                      'and put it under {}'.format(self.data_root))
                return False

    def recognize_all_data(self, test_area=5):
        ALL_FILES = self.getDataFiles('{}/indoor3d_sem_seg_hdf5_data/all_files.txt'.format(self.data_root))
        room_filelist = [line.rstrip() for line in open('{}/indoor3d_sem_seg_hdf5_data/room_filelist.txt'.format(self.data_root))]
        data_batch_list = []
        label_batch_list = []
        for h5_filename in ALL_FILES:
            data_batch, label_batch = self.loadDataFile('./{}/{}'.format(self.data_root,h5_filename))
            data_batch_list.append(data_batch)
            label_batch_list.append(label_batch)
        data_batches = np.concatenate(data_batch_list, 0)
        label_batches = np.concatenate(label_batch_list, 0)

        test_area = 'Area_' + str(test_area)
        train_idxs = []
        test_idxs = []
        for i, room_name in enumerate(room_filelist):
            if test_area in room_name:
                test_idxs.append(i)
            else:
                train_idxs.append(i)

        train_data = data_batches[train_idxs, ...]  # [16733, 4096, 9]
        train_label = label_batches[train_idxs]  # [16733, 4096]
        test_data = data_batches[test_idxs, ...]
        test_label = label_batches[test_idxs]
        print('train_data', train_data.shape, 'train_label', train_label.shape)
        print('test_data', test_data.shape, 'test_label', test_label.shape)
        return train_data, train_label, test_data, test_label

    def getDataFiles(self, list_filename):
        return [line.rstrip() for line in open(list_filename)]

    def loadDataFile(self, h5_filename):
        f = h5py.File(h5_filename,'r')
        data = f['data'][:]
        label = f['label'][:]
        return (data, label)

if __name__ == '__main__':

    dataset = indoor3d_Dataset(is_train=False, data_root='./')
    print(dataset.__len__())
