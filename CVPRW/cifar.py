from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from utils import download_url, check_integrity
import itertools
import random

im_thresh = 5000
type2_num = 150000
type1_num = 100000

def make_dataset(fdata):
    count = [0]*2
    newdataset = []
    for data in fdata:
        if not data[0][1] == data[1][1] :
            t = 0
            count[0]+=1
            item = (data[0][0],data[1][0],t)
            newdataset.append(item)
        
        else:
            t = 1
            count[1]+=1
            item = (data[0][0],data[1][0],t)
            newdataset.append(item)
    return newdataset
        

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_data1 = []
            self.train_data2 = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                # print(f,"\n")
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            ######################
            #### modification ####
            ######################

            cls_data = {}
            images = []

            for i in range(10):
                indexes = np.where(np.array(self.train_labels)==i)[0].tolist()
                cls_data[i] = [(im,i) for im in self.train_data[indexes]]
                self.train_data1.extend(list(itertools.combinations(cls_data[i][:200],2)))
                images.extend(cls_data[i][300:375])

            self.train_data2 = list(itertools.combinations(images,2))
            random.shuffle(self.train_data2)
            random.shuffle(self.train_data1)
            self.final_data = self.train_data1 + self.train_data2
            random.shuffle(self.final_data)
            self.trainset = make_dataset(self.final_data)
            print(len(self.trainset))
            ######################
            ######################

        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

            ######################
            #### modification ####
            ######################
            cls_data = {}
            images = []
            self.test_data1 = []
            self.test_data2 = []
            for i in range(10):
                indexes = np.where(np.array(self.test_labels)==i)[0].tolist()
                cls_data[i] = [(im,i) for im in self.test_data[indexes]]
                self.test_data1.extend(list(itertools.combinations(cls_data[i][:115],2)))
                images.extend(cls_data[i][200:240])
                
            self.test_data2 = list(itertools.combinations(images,2))
            random.shuffle(self.test_data2)
            random.shuffle(self.test_data1)
            self.final_test_data = self.test_data1 + self.test_data2
            random.shuffle(self.final_test_data)
            self.testset = make_dataset(self.final_test_data)
            ######################
            ######################

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.train:
            img1,img2 ,target = self.trainset[index][0],self.trainset[index][1],self.trainset[index][2]
            # print(img1.shape,img2.shape,target)
        else:
            img1,img2, target = self.testset[index][0],self.testset[index][1],self.testset[index][2]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
