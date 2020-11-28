import torch


def location_success_count(output, label, threshold=100):
    '''
    Simple location accuracy calculator based on distance
    '''
    output_locations = output[:, :, 0:2]
    label_locations = label[:, :, 0:2]

    distance = torch.sqrt(torch.sum((output_locations - label_locations)**2, axis = 2))
    
    count = torch.count_nonzero(distance < threshold)

    return count


def color_success_count(output, label, threshold = 100):

    output_colors = output[:, :, -3:]
    label_colors = label[:, :, -3:]

    distance = torch.sqrt(torch.sum((output_colors - label_colors)**2, axis = 2))
    count = torch.count_nonzero(distance < threshold)

    return count