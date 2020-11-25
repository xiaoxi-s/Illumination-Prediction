import time
import os
import copy
import data
import torch
import numpy as np

from datetime import datetime




def train_model(model, criterion, location_success_count, color_success_count, optimizer, scheduler, train_dataloader, test_dataloader, num_epochs=25):
    since = time.time()

    best_model_wts = None
    best_acc = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else ('cpu'))
    model.to(device)
    
    # to draw figures
    train_loss_epoch = np.zeros((num_epochs, 2))
    train_acc_epoch = np.zeros((num_epochs, 3))

    val_loss_epoch = np.zeros((num_epochs, 2))
    val_acc_epoch = np.zeros((num_epochs, 3))

    # weigths placed on the output (Ignore depth)
    weights = torch.tensor([1, 1, 0, 1, 1, 1, 1, 1, 1]).to(device)

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
            running_location_corrects = 0
            running_color_corrects = 0

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
                    loss = criterion(outputs, labels, weights)
                    running_location_corrects += location_success_count(outputs, labels)
                    running_color_corrects += color_success_count(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / len(dataloader)/dataloader.batch_size
            epoch_location_acc = running_location_corrects / len(dataloader)/3/dataloader.batch_size
            epoch_color_acc = running_color_corrects / len(dataloader)/3/dataloader.batch_size

            # record loss & acc during training
            if phase == 'train':
                train_loss_epoch[epoch][1] = epoch_loss
                train_acc_epoch[epoch][1] = epoch_location_acc
                train_acc_epoch[epoch][2] = epoch_color_acc

                train_loss_epoch[epoch][0] = epoch
                train_acc_epoch[epoch][0] = epoch
            else:
                val_loss_epoch[epoch][1] = epoch_loss
                val_acc_epoch[epoch][1] = epoch_location_acc
                val_acc_epoch[epoch][2] = epoch_color_acc

                val_loss_epoch[epoch][0] = epoch
                val_acc_epoch[epoch][0] = epoch

            print('{} Loss: {:.4f} Location Acc: {:.4f} Color Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_location_acc, epoch_color_acc))

            if epoch_location_acc > best_acc and phase == 'validation':
                best_model_wts = copy.deepcopy(model)
                best_acc = epoch_location_acc
    
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    if best_model_wts == 0 and num_epochs >= 25:
        raise TypeError("Accuracy Metric is Invalid")

    if num_epochs >= 25:
        torch.save(best_model_wts, os.path.join('checkpoint','model' + datetime.now().strftime("_%H_%M_%S_%d-%m-%Y")))

    return best_model_wts, train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch
