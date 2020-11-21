import time
import os
import copy
import data
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse


import dataset.transformer as transformer

from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms

from model.loss import average_difference_loss, location_success_count
from model.network import IlluminationPredictionNet
from dataset.dataset import EnvironmentJPGDataset


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    global train_dataloader
    global test_dataloader

    since = time.time()

    best_model_wts = None
    best_acc = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else ('cpu'))
    model.to(device)
    
    # to draw figures
    train_loss_epoch = np.zeros((num_epochs, 2))
    train_acc_epoch = np.zeros((num_epochs, 2))

    val_loss_epoch = np.zeros((num_epochs, 2))
    val_acc_epoch = np.zeros((num_epochs, 2))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_dataloader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = test_dataloader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for (i, data) in enumerate(dataloader):
                inputs = data[0].to(device)
                labels = data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = torch.reshape(outputs, (-1, 3, 9))
                    loss = criterion(outputs, labels)
                    running_corrects += location_success_count(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / len(dataloader)/dataloader.batch_size
            epoch_acc = running_corrects / len(dataloader)/3/dataloader.batch_size

            # record loss & acc during training
            if phase == 'train':
                train_loss_epoch[epoch][1] = epoch_loss
                train_acc_epoch[epoch][1] = epoch_acc

                train_loss_epoch[epoch][0] = epoch
                train_acc_epoch[epoch][0] = epoch
            else:
                val_loss_epoch[epoch][1] = epoch_loss
                val_acc_epoch[epoch][1] = epoch_acc

                val_loss_epoch[epoch][0] = epoch
                val_acc_epoch[epoch][0] = epoch

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if epoch_acc > best_acc and phase == 'validation':
                best_model_wts = copy.deepcopy(model)
                best_acc = epoch_acc
    
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    if best_model_wts is None and num_epochs >= 25:
        raise TypeError("Accuracy Metric is Invalid")

    if num_epochs >= 25:
        torch.save(best_model_wts, os.path.join('checkpoint','naive_model_with_activation' + datetime.now().strftime("_%H_%M_%S_%d-%m-%Y")))

    return best_model_wts, train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch


def plot_loss_acc(train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch):
    epoch_num = len(train_loss_epoch)

    # loss plot
    l1, = plt.plot(train_loss_epoch[:,0], train_loss_epoch[:, 1], color='blue')
    l2, = plt.plot(val_loss_epoch[:,0], val_loss_epoch[:, 1], color ='red')
    plt.legend(handles=[l1,l2],labels=['train','validation'],loc='best')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join("figures", "epoch-vs-squared_loss" + datetime.now().strftime("_%H-%M-%S_%d-%m-%Y") + ".png"))
    plt.close('all')

    # acc plot
    l1, = plt.plot(train_acc_epoch[:,0], train_acc_epoch[:,1], color='blue')
    l2, = plt.plot(val_acc_epoch[:,0], val_acc_epoch[:,1], color ='red')
    plt.legend(handles=[l1,l2],labels=['train','validation'],loc='best')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join("figures", "epoch-vs-accuracy"+ datetime.now().strftime("_%H-%M-%S_%d-%m-%Y") + ".png"))


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
    args = parser.parse_args()

    # args
    choice_of_optimizer = str.lower(str(args.optimizer))
    if choice_of_optimizer != 'sgd' and choice_of_optimizer != 'adam':
        raise TypeError('Optimizer Must be SGD or Adam')
    
    batch_size = args.batchsize

    learning_rate = args.learningrate
    sgd_momentum = args.momentum

    beta1 = args.beta1
    beta2 = args.beta2
    adam_beta = (beta1, beta2)
    adam_epsilon = args.epsilon

    height = args.height
    width = args.width
    #transformer.CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # dataset
    train_ds = EnvironmentJPGDataset(os.path.join('data', 'train_feature_matrix.npy'), os.path.join('data', 'train_label.npy'),\
        transform= transforms.Compose([
                                       transformer.ToTensor(),
                                       transformer.CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                       , augmentation=(1, 2))
    test_ds = EnvironmentJPGDataset(os.path.join('data', 'test_feature_matrix.npy'), os.path.join('data', 'test_label.npy'),\
        transform= transforms.Compose([
                                       transformer.ToTensor(),
                                       transformer.CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                       , augmentation=(1, 2))
    train_dataloader = DataLoader(train_ds, batch_size)
    test_dataloader = DataLoader(test_ds, batch_size)

    # model
    model = IlluminationPredictionNet()
    model.double()

    # optimizer
    if optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=sgd_momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=adam_beta, eps=adam_epsilon)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
    
    # train
    model, train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch = train_model(model, average_difference_loss, optimizer, scheduler, 1)

    plot_loss_acc(train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch)
