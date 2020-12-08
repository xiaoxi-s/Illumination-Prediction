import torch

def average_difference_loss(output, label, weights):
    '''
    Simple averaged squared loss between labels
    '''
    loss = torch.mean(((label - output)*weights)**2)

    return loss

def cos_difference_loss(output, label, weights):
    label[:, 0:2] = torch.cos(label[:, 0:2])
    output[:, 0:2] = torch.cos(output[:, 0:2])

    loss = torch.mean(((label - output)*weights)**2)

    return loss
