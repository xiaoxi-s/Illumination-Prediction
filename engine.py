'''
Interface for training and generating plots
'''

import os
import torch
import argparse
import matplotlib.pyplot as plt
import torch.optim as optim

import dataset.transformer as transformer

from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms

from train import train_model
from utils import plot_loss_acc
from dataset.dataset import EnvironmentJPGDataset
from model.network import IlluminationPredictionNet
from model.loss import average_difference_loss
from model.metric import location_success_count, color_success_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('-opt', '--optimizer', type=str,
                    help='choose SGD or Adam', default='sgd')
    parser.add_argument('-lr', '--learningrate', type=float, help='learning rate', default=0.0001)
    parser.add_argument('-mm', '--momentum', type=float, help='momentum for sgd', default=0.9)
    parser.add_argument('-b1', '--beta1', type=float, help='beta1 parameter for Adam', default=0.9)
    parser.add_argument('-b2', '--beta2', type=float, help='beta2 parameter for Adam', default=0.999)
    parser.add_argument('-e', '--epsilon', type=float, help='eps parameter for Adam', default=1e-8)
    parser.add_argument('-he', '--height', type=int, help = 'height of the input image', default=400)
    parser.add_argument('-w', '--width', type=int, help = 'width of the input image', default=900)
    parser.add_argument('-bs', '--batchsize', type=int, help='batch size', default=16)
    parser.add_argument('-epoch', '--epoch', type=int, help='training epoch', default = 50)
    parser.add_argument('-ag1', '--augmentation_row_num', type=int, help='number of rows for data augmentation', default=1)
    parser.add_argument('-ag2', '--augmentation_col_num', type=int, help='number of columns for data augmentation', default=1)
    parser.add_argument('-ft', '-finetune', type=int, help='whether fix the feature extractor', default=1)
    args = parser.parse_args()

    # args
    choice_of_optimizer = str.lower(str(args.optimizer))
    if choice_of_optimizer != 'sgd' and choice_of_optimizer != 'adam':
        raise TypeError('Optimizer Must be SGD or Adam')
    fine_tune = args.finetune
    if fine_tune == 0:
        fine_tune = False
    else:
        fine_tune = True

    batch_size = args.batchsize
    epoch = args.epoch

    learning_rate = args.learningrate
    sgd_momentum = args.momentum

    beta1 = args.beta1
    beta2 = args.beta2
    adam_beta = (beta1, beta2)
    adam_epsilon = args.epsilon

    height = args.height
    width = args.width

    augmentation_param = (args.augmentation_row_num, args.augmentation_col_num)
    
    # dataset
    train_ds = EnvironmentJPGDataset(os.path.join('data', 'train_feature_matrix.npy'), os.path.join('data', 'train_label.npy'),\
        transform= transforms.Compose([
                                       transformer.ToTensor(),
                                       transformer.CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                       , augmentation=augmentation_param)
    test_ds = EnvironmentJPGDataset(os.path.join('data', 'test_feature_matrix.npy'), os.path.join('data', 'test_label.npy'),\
        transform= transforms.Compose([
                                       transformer.ToTensor(),
                                       transformer.CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                       , augmentation=augmentation_param)
    train_dataloader = DataLoader(train_ds, batch_size)
    test_dataloader = DataLoader(test_ds, batch_size)

    # model
    model = IlluminationPredictionNet(fine_tune)
    model.double()

    # optimizer
    if optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=sgd_momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=adam_beta, eps=adam_epsilon)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
    weights = [20, 20, 0, 1, 1, 1, 1, 1, 1]
    weights = torch.nn.functional.normalize(torch.Tensor(weights), dim = 0)
    # train
    model, train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch = train_model(
        model, average_difference_loss, location_success_count, color_success_count, \
        optimizer, scheduler, train_dataloader, test_dataloader, weights, epoch)

    plot_loss_acc(train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch)

    if epoch > 1:
        torch.save(model, os.path.join('checkpoint', datetime.now().strftime("_%d-%m-%Y_%H_%M_%S") + 'model'))

