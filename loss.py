import torch

def average_difference_loss(output, label):
    '''
    Simple averaged squared loss between labels
    '''
    num = output.shape[0] * 3
    loss = torch.mean((label - output)**2)/num

    return loss


def location_success_count(output, label, threshold=100):
    '''
    Simple location accuracy calculator based on distance
    '''
    output_locations = output[:, :, 0:2]
    label_locations = label[:, :, 0:2]

    distance = torch.sum((output_locations - label_locations)**2, axis = 2)
    
    count = torch.count_nonzero(distance < 100)

    return count

