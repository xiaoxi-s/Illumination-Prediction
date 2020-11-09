import os
import torch
import torchvision.models as models
import transformer 

from dataset import EnvironmentJPGDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from network import IlluminationPredictionNet

def test_dataset():
    ds = EnvironmentJPGDataset(os.path.join('data', 'labeled_images.npy'), os.path.join('data', 'labels.npy'))

    for i in range(1, 10):
        img = ds[i]['images']
        assert img.shape == (400, 900, 3)


def test_transformer():
    ds = EnvironmentJPGDataset(os.path.join('data', 'labeled_images.npy'), os.path.join('data', 'labels.npy'),\
        transform= transforms.Compose([transformer.Rescale((256, 256)),
                                       transformer.ToTensor()]))


def test_run_in_vgg11():
    ds = EnvironmentJPGDataset(os.path.join('data', 'labeled_images.npy'), os.path.join('data', 'labels.npy'),\
        transform= transforms.Compose([transformer.Rescale((224, 224)),
                                       transformer.ToTensor()]))
    dataloader = DataLoader(ds, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else ('cpu'))

    model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11', pretrained=True)
    model.double()
    model.to(device)
    
    with torch.no_grad():
        for (i, data) in enumerate(dataloader):
            # inputs,_ = data

            img = data['images']

            img = img.to(device)
            output = model(img)


def test_run_in_illumination_prediction_net():
    model = IlluminationPredictionNet()
    ds = EnvironmentJPGDataset(os.path.join('data', 'labeled_images.npy'), os.path.join('data', 'labels.npy'),\
    transform= transforms.Compose([transformer.Rescale((224, 224)),
                                       transformer.ToTensor()]))
    dataloader = DataLoader(ds, 1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('cpu'))

    model.double()
    model.to(device)

    with torch.no_grad():
        for (i, data) in enumerate(dataloader):
            # inputs,_ = data

            img = data['images']
            img = img.to(device)
            output = model(img)


if __name__ == '__main__':
    #test_dataset()
    #test_transformer()
    #test_run_in_vgg11()
    #test_run_in_illumination_prediction_net()