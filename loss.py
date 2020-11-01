import torch

def average_difference_loss(output, label):
    # label = torch.vstack((label, label, label, label, label, label))
    
    # output2 = output[[0, 2, 1]]
    # output3 = output[[2, 0, 1]]
    # output4 = output[[1, 0 ,2]]
    # output5 = output[[1, 2, 0]]
    # output6 = output[[2, 1, 0]]

    # output = torch.vstack((output, output2, output3, output4, output5, output6))
    loss = torch.mean((label - output)**2)

    return loss
