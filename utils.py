import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from datetime import datetime


def convert_exr_and_write(exr, dest_path, img_format, default_height = 400, default_width = 900):
    im=cv2.imread(exr,-1)
    height, width = im.shape[:2]

    # convert to jpg
    if img_format == 'jpg':
        # tonemap = cv2.createTonemap(gamma=1)
        # im = tonemap.process(im)
        # im = im[0:height-600,:]

        im = cv2.normalize(im, None, alpha=0, beta = 20000, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        im = np.uint16(im)
        
        # im = cv2.resize(im, (default_width, default_height))
        
        cv2.imwrite(dest_path, im)
    elif img_format == 'exr': # convert to exr
        tonemap = cv2.createTonemap(gamma=1)
        im = tonemap.process(im)
        im = im[0:height-600,:]

        im = cv2.resize(im, (default_width, default_height))
        
        cv2.imwrite(dest_path, im)
    else:
        raise TypeError('Supported format jpg and exr')


def imshow(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)


def plot_loss_acc(train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch):
    epoch_num = len(train_loss_epoch)

    # loss plot
    l1, = plt.plot(train_loss_epoch[:,0], train_loss_epoch[:, 1], color='blue')
    l2, = plt.plot(val_loss_epoch[:,0], val_loss_epoch[:, 1], color ='red')
    plt.legend(handles=[l1,l2],labels=['train','validation'],loc='best')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join("figures", datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "epoch-loss" + ".png"))
    plt.close('all')

    # location acc plot
    l1, = plt.plot(train_acc_epoch[:,0], train_acc_epoch[:,1], color='blue')
    l2, = plt.plot(val_acc_epoch[:,0], val_acc_epoch[:,1], color ='red')
    plt.legend(handles=[l1,l2],labels=['train','validation'],loc='best')
    plt.title('Location Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('location accuracy')
    plt.savefig(os.path.join("figures", datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "epoch-loc-acc" + ".png"))
    plt.close('all')
    
    # color acc plot
    l1, = plt.plot(train_acc_epoch[:,0], train_acc_epoch[:,2], color='blue')
    l2, = plt.plot(val_acc_epoch[:,0], val_acc_epoch[:,2], color ='red')
    plt.legend(handles=[l1,l2],labels=['train','validation'],loc='best')
    plt.title('Color Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('color accuracy')
    plt.savefig(os.path.join("figures", datetime.now().strftime("%d-%m-%Y_%H-%M-%S") +  "epoch-color-acc"+ ".png"))


def load_model(model_path_with_name):
    model = torch.load(model_path_with_name)

    return model


def patch_data_from_npz(folder_path, dest_path):
    npz_names = os.listdir(os.path.join(folder_path))
    npz_names.sort()

    train_matrix = None
    train_labels = None

    test_matrix = None
    test_labels = None

    for npz in npz_names:

        if os.path.isdir(os.path.join(folder_path, npz)):
            continue

        dataset_dict = np.load(os.path.join(folder_path, npz))

        if train_matrix is None:
            train_matrix = dataset_dict['train_matrix']
            train_labels = dataset_dict['train_labels']

            test_matrix = dataset_dict['test_matrix']
            test_labels = dataset_dict['test_labels']
        else:
            train_matrix = np.concatenate((train_matrix, dataset_dict['train_matrix']), axis=0)
            train_labels = np.concatenate((train_labels, dataset_dict['train_labels']), axis=0)

            test_matrix = np.concatenate((test_matrix, dataset_dict['test_matrix']), axis=0)
            test_labels = np.concatenate((test_labels, dataset_dict['test_labels']), axis=0)
    
    np.save(os.path.join(dest_path, 'train_feature_matrix.npy'), train_matrix, allow_pickle=True)
    np.save(os.path.join(dest_path, 'train_label.npy'), train_labels, allow_pickle=True)

    np.save(os.path.join(dest_path, 'test_feature_matrix.npy'), test_matrix, allow_pickle=True)
    np.save(os.path.join(dest_path, 'test_label.npy'), test_labels, allow_pickle=True)


def patch_data_from_npz(folder_path, dest_path):
    raise RuntimeError("Patch it mannually")