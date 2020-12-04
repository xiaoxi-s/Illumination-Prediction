import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader


class EnvironmentJPGDataset(Dataset):
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

        # agumentation part
        if augmentation is not None:
            ratio = int(augmentation[0] * augmentation[1])
            row_num = int(augmentation[0])
            col_num = int(augmentation[1])

            temp_env_frame = np.zeros((int(ratio * size), height//row_num, width//col_num, channel))
            temp_labels = np.zeros((int(ratio * size), N, m))

            # for each instance
            for i in range(0, len(self.environment_frame)):
                # seperate it into row_num * col_num sub-instances
                for j in range(0, row_num):
                    for k in range(0, col_num):

                        temp_env_frame[int(ratio*i) + j*col_num + k] = self.environment_frame[i][ j*height//row_num : (j+1)*height//row_num, k*width//col_num:(k+1)*width//col_num, :]
                        temp_labels[int(ratio*i) + j*col_num+k] = self.labels[i]

            self.environment_frame = temp_env_frame
            self.labels = temp_labels

        if self.transform:
            size = len(self.environment_frame)
            temp_env_frame = np.zeros((size, channel, height//row_num, width//col_num))
            temp_labels = np.zeros((size, N, m))

            for i in range(0, len(self.environment_frame)):
                temp_env_frame[i], temp_labels[i] = self.transform([self.environment_frame[i], self.labels[i]])

            self.environment_frame = temp_env_frame
            self.labels = temp_labels

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
