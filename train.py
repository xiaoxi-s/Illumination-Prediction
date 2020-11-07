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

from loss import average_difference_loss
from network import IlluminationPredictionNet
from dataset import EnvironmentJPGDataset


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    global dataloader
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
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
            else:
                model.eval()   # Set model to evaluate mode

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

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    torch.save(model, os.path.join('checkpoint','naive_model' + datetime.now().strftime("_%H:%M:%S_%d-%m-%Y")))
    # load best model weights

    return model


if __name__ == '__main__':
    ds = EnvironmentJPGDataset(os.path.join('data', 'labeled_images.npy'), os.path.join('data', 'labels.npy'),\
        transform= transforms.Compose([transformer.Rescale((224, 224)),
                                       transformer.ToTensor()]))
    dataloader = DataLoader(ds, 1)

    model = IlluminationPredictionNet()
    model.double()

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model(model, average_difference_loss, optimizer, scheduler, 25)