import torch
from torchsummary import summary
model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)

print(model)

