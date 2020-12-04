import os
import torch
import cv2
import torchvision.models as models
import dataset.transformer 
import utils
import dataset.preprocess as pp
import dataset.imagecropper 
import math
import numpy as np

from dataset.dataset import EnvironmentJPGDataset
from dataset.dataset import EnvironmentEXRDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from model.network import IlluminationPredictionNet

def test_dataset():
    ds = EnvironmentJPGDataset(os.path.join('data', 'labeled_images.npy'), os.path.join('data', 'labels.npy'))

    for i in range(1, 10):
        img = ds[i]['images']
        assert img.shape == (400, 900, 3)


def test_transformer():
    ds = EnvironmentEXRDataset(os.path.join('data', 'labeled_images.npy'), os.path.join('data', 'labels.npy'),\
        transform= transforms.Compose([transformer.Rescale((256, 256)),
                                       transformer.ToTensor()]))


def test_dataset_new():
    ds = EnvironmentEXRDataset('./data/train_feature_matrix.npy', './data/train_label.npy')

     

def test_new_labeler():
    labeler = pp.EXRLabeler()
    img = cv2.imread("./data/exr/02.exr", cv2.IMREAD_UNCHANGED)

    labels = labeler.generate_labels(img)
    print((labels))

def test_sphericalSystem():
    img=cv2.imread('./data/EnvironmentMapTesting.jpg', -1)
    handle = dataset.imagecropper.imagecropper(img)
    img_new = handle.generate_image(math.pi, math.pi/2, math.pi/2.,1080, 720)
    cv2.imshow("old",img_new)
    cv2.waitKey(0)

def test_dataGen():
    matrix_generator = pp.DataGenerator()
    matrix_generator.generate_feature_and_label_new("./data/exr/", './data', False)


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
    test_dataGen()