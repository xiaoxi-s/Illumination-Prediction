import torch
import numpy as np

from skimage import transform

class Normalize(object):
    '''Normalize image matrix'''

    def __init__(self):
        pass

    def __call__(self, input):
        image, labels = input[0], input[1]
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * [new_w / w, new_h / h]

        return img, labels

''' source code from pytorch '''

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, input):
        image, labels = input[0], input[1]
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * [int(new_w / w), int(new_h / h)]

        return img, labels


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, input):
        image, labels = input[0], input[1]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(labels)


''' Normalize Images in torch.tensor'''
class CustomNormalize(object):
    def __init__(self, mean, std, scaling_factor=1):
        self.mean = torch.from_numpy(np.resize(mean, (3, 1, 1)))
        self.std = torch.from_numpy(np.resize(std, (3, 1, 1)))
        self.scaling_factor = scaling_factor
        
    def __call__(self, input):
        image, labels = input[0], input[1]
        output = (self.scaling_factor*image - self.mean) / self.std

        return output, labels

