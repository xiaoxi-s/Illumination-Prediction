import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime


def convert_exr_and_write(exr, dest_path, img_format, default_height = 400, default_width = 900):
    im=cv2.imread(exr,-1)
    height, width = im.shape[:2]

    # convert to jpg
    if img_format == 'jpg':
        tonemap = cv2.createTonemap(gamma=1)
        im = tonemap.process(im)
        im = im[0:height-600,:]

        im = cv2.normalize(im, None, alpha=0, beta=20000, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im=np.uint16(im)
        
        im = cv2.resize(im, (default_width, default_height))
        
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
