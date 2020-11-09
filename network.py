import torch
from torch import nn
from torchsummary import summary

class IlluminationPredictionNet(nn.Module):
    def __init__(self):
        super(IlluminationPredictionNet, self).__init__()
        self.dense121_fixed = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
        for param in self.dense121_fixed.parameters():
            param.requires_grad = False

        in_features = self.dense121_fixed.classifier.in_features

        self.dense121_fixed.classifier = nn.Linear(in_features, 512)
        self.output_layer = nn.Linear(512, 9)


    def forward(self, input):
        output = self.dense121_fixed(input)
        output = self.output_layer(output)

        return output




    #print(model)


