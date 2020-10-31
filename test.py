import os
import torch
import torchvision.models as models

from torch.utils.data import DataLoader
from dataloader import EnvironmentJPGDataset
from torchvision import transforms
import transformer 

def test_dataset():
    ds = EnvironmentJPGDataset(os.path.join('data', 'labeled_images.npy'), os.path.join('data', 'labels.npy'))

    for i in range(1, 10):
        img = ds[i]['images']
        print('Size {} of {} th sample'.format(img.shape, i))
        print(img.shape)


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
            
            print(output.shape)
            print(torch.argmax(output))

if __name__ == '__main__':
    # test_dataset()
    # test_transformer()
    test_run_in_vgg11()