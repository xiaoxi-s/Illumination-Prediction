import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

import dataset.transformer as transformer

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime

from utils import load_model
from dataset.transformer import CustomNormalize
from dataset.dataset import EnvironmentJPGDataset
from model.metric import location_success_count, color_success_count


def evaluate(model, loc_thresh, color_thresh, dataloader, N, num_of_param, verbose=False):

    if (len(loc_thresh) > len(color_thresh)):
        larger = len(loc_thresh)
    else:
        larger = len(color_thresh)
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('cpu'))
    loc_thresh_vs_acc = np.zeros((len(loc_thresh), 2))
    color_thresh_vs_acc = np.zeros((len(color_thresh), 2))
    
    since =time.time()

    for (i, data) in enumerate(dataloader):
        inputs = data[0].to(device)
        labels = data[1].to(device)
        print('{} Iteration: '.format(i))
        # evaluation
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = torch.reshape(outputs, (-1, N, num_of_param))
            for j in range(0, larger):
                # statistics
                if j < len(loc_thresh):
                    loc_thresh_vs_acc[j][0] = loc_thresh[j]
                    loc_thresh_vs_acc[j][1] += int(location_success_count(outputs, labels, loc_thresh[j]))
                    if verbose:
                        print('Loc Thresh: {}, Loc success count: {}'.format(loc_thresh[j], loc_thresh_vs_acc[j][1]))
                if j < len(color_thresh):
                    color_thresh_vs_acc[j][0] = color_thresh[j]
                    color_thresh_vs_acc[j][1] += int(color_success_count(outputs, labels, color_thresh[j]))
                    if verbose:
                        print('Color Thresh: {}, Color success count: {}'.format(color_thresh[j], color_thresh_vs_acc[j][1]))
    
    # divide by number of samples
    color_thresh_vs_acc[:, 1] = color_thresh_vs_acc[:, 1]/len(dataloader)/N/dataloader.batch_size
    color_thresh_vs_acc[:, 0] = color_thresh_vs_acc[:, 0]/np.amax(color_thresh_vs_acc[:, 0])
    loc_thresh_vs_acc[:, 1] = loc_thresh_vs_acc[:, 1] / len(dataloader)/N/dataloader.batch_size
    loc_thresh_vs_acc[:, 0] = loc_thresh_vs_acc[:, 0]/np.amax(loc_thresh_vs_acc[:, 0])
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return loc_thresh_vs_acc, color_thresh_vs_acc


def plot_threshold_and_accuracy(loc, acc):
    l1, = plt.plot(loc[:,0], loc[:, 1], color='blue')
    l2, = plt.plot(acc[:,0], acc[:, 1], color ='red')
    plt.legend(handles=[l1,l2],labels=['loc-acc','color-acc'],loc='best')
    plt.title('Training and Validation Loss')
    plt.xlabel('param normalized to (0, 1]')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join("figures", "sensitivity-analysis", datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "epoch-loss" + ".png"))
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate figures for sensitivity analysis')
    # model
    parser.add_argument('-dp', '--data_path', type=str, help='path to folder containing data', default='data')
    parser.add_argument('-p' ,'--model_path_with_name', type=str, help='path to model to evaluate')
    parser.add_argument('-bs', '--batchsize', type=int, help='batch size', default=16)
    parser.add_argument('-N', '--N', type=int, help='num of light sources', default=6)
    parser.add_argument('-np', '--num_of_param', type=int, help='num of param for each light source', default=5)
    parser.add_argument('-v', '--verbose', type=int, help='whether output info', default=0)

    # threshold range & step size
    parser.add_argument('-lts', '--location_thresh_start', type=float, help='start threshold for loc acc', default=10)
    parser.add_argument('-lte', '--location_thresh_end', type=float, help='end threshold for loc acc', default=np.sqrt(400*400 + 900*900))
    parser.add_argument('-ltn', '--location_thresh_num', type=int, help='num of thresholds within range', default=20)

    parser.add_argument('-cts', '--color_thresh_start', type=float, help='start threshold for color acc', default=10)
    parser.add_argument('-cte', '--color_thresh_end', type=float, help='end threshold for color acc', default=np.sqrt(3)*255)
    parser.add_argument('-ctn', '--color_thresh_num', type=int, help='num of thresholds within range', default=20)

    args = parser.parse_args()
    data_path = args.data_path
    batch_size = args.batchsize
    model_path_with_name = args.model_path_with_name
    N = args.N
    num_of_param = args.num_of_param
    verbose = False if args.verbose == 0 else True

    # thresholds
    location_thresh_start = args.location_thresh_start
    location_thresh_end = args.location_thresh_end
    location_thresh_num = args.location_thresh_num

    color_thresh_start = args.color_thresh_start
    color_thresh_end = args.color_thresh_end
    color_thresh_num = args.color_thresh_num

    # thresholods in torch tensor
    loc_thresh = np.linspace(location_thresh_start, location_thresh_end, num=location_thresh_num)
    color_thresh = np.linspace(color_thresh_start, color_thresh_end, num=color_thresh_num)

    # prepare dataset    
    test_ds = EnvironmentJPGDataset(os.path.join(data_path, 'test_feature_matrix.npy'), os.path.join(data_path, 'test_label.npy'),\
        transform= transforms.Compose([transformer.CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), model_type='f')
    test_dataloader = DataLoader(test_ds, batch_size, shuffle=False)

    # load model
    print("Model to be evaluated: {}".format(model_path_with_name))
    model = load_model(model_path_with_name)
    
    loc_thresh_vs_acc, color_thresh_vs_acc = evaluate(model, loc_thresh, color_thresh, test_dataloader, N, num_of_param, verbose)

    plot_threshold_and_accuracy(loc_thresh_vs_acc, color_thresh_vs_acc)

    np.save('figures/sensitivity-analysis/loc_thresh_vs_acc.npy', loc_thresh_vs_acc)
    np.save('figures/sensitivity-analysis/color_thresh_vs_acc.npy', color_thresh_vs_acc)
