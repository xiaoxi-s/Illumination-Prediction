import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

class EnvironmentJPGDataset(Dataset):
    def __init__(self, img_npy_file, label_npy_file, transform=None):
        """
        Args:
            img_npy_file (string): Path to the image npy file.
            label_npy_file: Path to the image label npy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.environment_frame = np.load(img_npy_file, allow_pickle=True)
        self.labels = np.load(label_npy_file, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.environment_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'images': self.environment_frame[idx], 'labels': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample



