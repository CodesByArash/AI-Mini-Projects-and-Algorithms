import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from utils import *

class SignDigitDataset(Dataset):

    def __init__(self, root_dir='data/', h5_name='train_signs.h5',train=True, transform=None, initialization=None):
        self.transform = transform
        self.train = train

        # h5 file related attributes
        self.h5_path = os.path.join(root_dir, h5_name)
        key = 'train_set' if self.train else 'test_set'
        self.dataset_images = np.array(self._read_data(self.h5_path)[key+'_x'])
        self.dataset_labels = np.array(self._read_data(self.h5_path)[key + '_y'])


    def __len__(self):
        return len(self.dataset_images)


    def __getitem__(self, index):
        #####################################################################################
        # First transform the image                                                         #
        # And then use utils.get_one_hot() to create the proper label for model             #
        #####################################################################################
        img = None
        label = None
        # my code
        if self.transform is not None:
            img = self.transform(self.dataset_images[index])
        else :
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(self.dataset_images[index])
        # crossentropy does not need one hot 👇️
        # label = get_one_hot(self.dataset_labels[index],10)
        label = self.dataset_labels[index]

        return img,label

    def _read_data(self, h5_path):
        dataset = h5py.File(h5_path, "r")

        return dataset