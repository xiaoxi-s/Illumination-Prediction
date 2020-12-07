import torch
import numpy as np

def location_success_count(output, label, threshold=100):
    '''
    Simple location accuracy calculator based on distance
    '''
    output_locations = output[:, :, 0:2]
    label_locations = label[:, :, 0:2]

    distance = torch.sqrt(torch.sum((output_locations - label_locations)**2, axis = 2))
   
    bool_mat =(distance < threshold).cpu()
    count = np.count_nonzero(bool_mat)

    return count


def color_success_count(output, label, threshold = 100):

    output_colors = output[:, :, -3:]
    label_colors = label[:, :, -3:]

    distance = torch.sqrt(torch.sum((output_colors - label_colors)**2, axis = 2))
    bool_mat =(distance < threshold).cpu()
    count = np.count_nonzero(bool_mat)

    return count
