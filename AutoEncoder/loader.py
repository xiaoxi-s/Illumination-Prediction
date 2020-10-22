# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Henrique Weber
# LVSN, Universite Laval and Institute National d'Optique
# Email: henrique.weber.1@ulaval.ca
# Copyright (c) 2018
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import numpy as np
import torch


# This belongs to pytorch_toolbox
# from pytorch_toolbox.loader_base import LoaderBase

from torch.utils.data.dataset import Dataset
from abc import ABCMeta, abstractmethod
from tools.utils import load_hdr
class LoaderBase(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, root, data_transforms=[], target_transforms=[]):
        imgs = self.make_dataset(root)
        self.root = root
        self.imgs = imgs
        self.data_transforms = data_transforms
        self.target_transforms = target_transforms

    @abstractmethod
    def make_dataset(self, dir):
        """
        Should return a list of (data, target)
        :param path:
        :param class_to_idx:
        :return:
        """
        pass

    @abstractmethod
    def from_index(self, index):
        """
        should return a tuple : (data, target)
        :param index:
        :return:
        """
        pass

    def __getitem__(self, index):
        data, target, info = self.from_index(index)
        if not isinstance(data, list):
            data = [data]

        for i, transform in enumerate(self.data_transforms):
            if transform is not None:
                data[i] = transform(data[i])
        for i, transform in enumerate(self.target_transforms):
            if transform is not None:
                target[i] = transform(target[i])
        return data, target, info

    def __len__(self):
        return len(self.imgs)


class AutoEncoderDataset(LoaderBase):

    def __init__(self, root, transform=[], target_transform=[]):
        self.images = self.make_dataset(root)
        super().__init__(root, transform, target_transform)

    def make_dataset(self, root):
        dir = os.path.expanduser(root)
        images = []
        files = [x for x in os.listdir(dir) if x.endswith('.exr')]
        for file in files:
            path = os.path.join(dir, file)
            images.append(path)

        return images

    def __len__(self):
        return len(self.images)

    def get_name(self, index):
        return self.images[index]

    def from_index(self, index):

        path = self.images[index]
        image = [load_hdr(path)]
        index = torch.from_numpy(np.array([index]))
        info = {'path': os.path.basename(os.path.normpath(path))}
        return image, index, info


# This function is initially imported in the package implemented by the author
# from learning_indoor_lighting.tools.utils import load_hdr
# def load_hdr(path):
#     """
#     Loads an HDR file with RGB channels
#     :param path: file location
#     :return: HDR image
#     """
#     pt = Imath.PixelType(Imath.PixelType.FLOAT)
#     rgb_img_openexr = OpenEXR.InputFile(path)
#     rgb_img = rgb_img_openexr.header()['dataWindow']
#     size_img = (rgb_img.max.x - rgb_img.min.x + 1, rgb_img.max.y - rgb_img.min.y + 1)

#     redstr = rgb_img_openexr.channel('R', pt)
#     red = np.fromstring(redstr, dtype=np.float32)
#     red.shape = (size_img[1], size_img[0])

#     greenstr = rgb_img_openexr.channel('G', pt)
#     green = np.fromstring(greenstr, dtype=np.float32)
#     green.shape = (size_img[1], size_img[0])

#     bluestr = rgb_img_openexr.channel('B', pt)
#     blue = np.fromstring(bluestr, dtype=np.float32)
#     blue.shape = (size_img[1], size_img[0])

#     hdr_img = np.dstack((red, green, blue))

#     return hdr_img
