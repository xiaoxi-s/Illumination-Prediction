import torch
from torch import nn
from torchsummary import summary

class IlluminationPredictionNet(nn.Module):
    def __init__(self):
        # required by the pytorch api
        super(IlluminationPredictionNet, self).__init__()

        # feature extractor
        #self.wrn50_2 = torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2', pretrained=True)
        self.dense121_fixed = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
        for param in self.dense121_fixed.parameters():
            param.requires_grad = False

        in_features = self.dense121_fixed.classifier.in_features
        # out_features = self.wrn50_2.fc.out_features
        # linear + activation + linear (the out put)
        self.dense121_fixed.classifier = nn.Linear(in_features, 512)

        self.activation = nn.ReLU(512)
        self.output_layer = nn.Linear(512, 27)


    def forward(self, input):
        output = self.dense121_fixed(input)
        output = self.activation(output)
        output = self.output_layer(output)

        return output
