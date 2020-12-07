import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader


class EnvironmentJPGDataset(Dataset):
    def __init__(self, img_npy_file, label_npy_file, model_type,transform=None, augmentation=None):
        """
        Args:
            img_npy_file (string): Path to the image npy file.
            label_npy_file: Path to the image label npy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            augmentation: a tuple of 2 dims. First is the number of rows; second 
                is the number of colomns.
        """
        self.environment_frame = np.load(img_npy_file, allow_pickle=True)
        self.labels = np.load(label_npy_file, allow_pickle=True)
        assert len(self.environment_frame) == len(self.labels)

        self.transform = transform
        self.environment_frame = self.environment_frame.transpose((0, 3, 1, 2))
        self.environment_frame = torch.from_numpy(self.environment_frame) 
        self.labels = torch.from_numpy(self.labels)
        if model_type == 'f':
            self.environment_frame.float()
            self.labels.float()
        else:
            self.environment_frame.double()
            self.labels.double()
        size, height, width, channel = self.environment_frame.shape
        _, N, m = self.labels.shape
        ratio = None

        if self.transform:
            for i in range(0, len(self.environment_frame)):
                self.environment_frame[i], self.labels[i] = self.transform([self.environment_frame[i], self.labels[i]])

    def __len__(self):
        return len(self.environment_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.environment_frame[idx], self.labels[idx]



class EnvironmentEXRDataset(Dataset):
    def __init__(self, img_npy_file, label_npy_file, transform=None, augmentation=None):
        """
        Args:
            img_npy_file (string): Path to the image npy file.
            label_npy_file: Path to the image label npy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            augmentation: a tuple of 2 dims. First is the number of rows; second 
                is the number of colomns.
        """
        self.environment_frame = np.load(img_npy_file, allow_pickle=True)
        self.labels = np.load(label_npy_file, allow_pickle=True)
        self.transform = transform

        size, height, width, channel = self.environment_frame.shape
        _, N, m = self.labels.shape
        ratio = None



    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
