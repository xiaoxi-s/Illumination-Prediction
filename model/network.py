import torch
from torch import nn
from torchsummary import summary

class IlluminationPredictionNet(nn.Module):
    def __init__(self):
        # required by the pytorch api
        super(IlluminationPredictionNet, self).__init__()

        # feature extractor
        self.dense121_fixed = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
        for param in self.dense121_fixed.parameters():
            param.requires_grad = False

        in_features = self.dense121_fixed.classifier.in_features

        # linear + activation + linear (the out put)
        self.dense121_fixed.classifier = nn.Linear(in_features, 512)
        self.output_layer = nn.Linear(512, 27)


    def forward(self, input):
        output = self.dense121_fixed(input)
        output = self.output_layer(output)

        return output




    #print(model)


