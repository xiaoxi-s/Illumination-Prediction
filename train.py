import time
import os
import copy
import data
import torch
import torch.optim as optim

import transformer

from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms

from Training.loss import average_difference_loss, location_success_count
from Training.network import IlluminationPredictionNet
from Training.dataset import EnvironmentJPGDataset


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    global train_dataloader
    global test_dataloader

    since = time.time()

    best_model_wts = None
    best_acc = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else ('cpu'))
    model.to(device)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
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
                inputs = data['images'].to(device)
                labels = data['labels'].to(device)

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
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_corrects / len(dataloader)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if epoch_acc > best_acc and phase == 'validation':
                best_model_wts = copy.deepcopy(model)
                best_acc = acc
    
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    if best_model_wts is None:
        raise TypeError("Accuracy Metric is Invalid")

    torch.save(best_model_wts, os.path.join('checkpoint','naive_model' + datetime.now().strftime("_%H:%M:%S_%d-%m-%Y")))
    # load best model weights

    return model


if __name__ == '__main__':
    train_ds = EnvironmentJPGDataset(os.path.join('data', 'train_feature_matrix.npy'), os.path.join('data', 'train_label.npy'),\
        transform= transforms.Compose([transformer.Rescale((224, 224)),
                                       transformer.ToTensor()]))
    test_ds = EnvironmentJPGDataset(os.path.join('data', 'test_feature_matrix.npy'), os.path.join('data', 'test_label.npy'),\
        transform= transforms.Compose([transformer.Rescale((224, 224)),
                                       transformer.ToTensor()]))

    train_dataloader = DataLoader(train_ds, 1)
    test_dataloader = DataLoader(test_ds, 1)

    model = IlluminationPredictionNet()
    model.double()

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model(model, average_difference_loss, optimizer, scheduler, 25)