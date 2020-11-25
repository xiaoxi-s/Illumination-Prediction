import torch

def average_difference_loss(output, label, weights):
    '''
    Simple averaged squared loss between labels
    '''
    test = (label - output)*weights
    loss = torch.mean(((label - output)*weights)**2)

    return loss