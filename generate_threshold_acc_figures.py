import torch
import argparse
import matplotlib as plt
import numpy as np

from dataset import EnvironmentJPGDataset


def evaluate(model, loc_thresh, color_thresh, dataloader):

    if (len(loc_thresh) > len(color_thresh)):
        larger = len(loc_thresh)
    else:
        larger = len(color_thresh)
    
    loc_thresh_vs_acc = np.zeros((len(loc_thresh, 2)))
    color_thresh_vs_acc = np.zeros((len(color_thresh), 2))
    
    for (i, data) in enumerate(dataloader):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        # evaluation
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = torch.reshape(outputs, (-1, 3, 9))
            loss = criterion(outputs, labels, weights)
            for j in range(0, larger):
                # statistics
                if j < len(loc_thresh):
                    loc_thresh_vs_acc[j][0] = loc_thresh[j]
                    loc_thresh_vs_acc[j][1] = int(location_success_count(outputs, labels), loc_thresh[j])
                if j < len(color_thresh):
                    color_thresh_vs_acc[j][0] = color_thresh[j]
                    color_thresh_vs_acc[j][1] = int(color_success_count(outputs, labels), color_thresh[j])
    
    # divide by number of samples
    color_thresh_vs_acc[:, 1] = color_thresh_vs_acc[:, 1]/len(dataloader)/3/dataloader.batch_size
    loc_thresh_vs_acc[:, 1] = loc_thresh_vs_acc/ len(dataloader)/3/dataloader.batch_size
    
    return loc_thresh_vs_acc, color_thresh_vs_acc


def plot_threshold_and_accuracy(loc, acc):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate figures for sensitivity analysis')
    # model
    parser.add_argument('-p' ,'--model_path', type=str, help='path to model to evaluate')
    parser.add_argument('-bs', '--batchsize', type=int, help='batch size', default=16)

    # threshold range & step size
    parser.add_argument('-lts', '--location_thresh_start', type=float, help='start threshold for loc acc', default=10)
    parser.add_argument('-lte', '--location_thresh_end', type=float, help='end threshold for loc acc', default=np.sqrt(400*400 + 900*900)))
    parser.add_argument('-ltss', '--location_thresh_step_size', type = float, help='step size for loc acc', default=10)

    parser.add_argument('-cts', '--color_thresh_start', type=float, help='start threshold for color acc', default=10)
    parser.add_argument('-cte', '--color_thresh_end', type=float, help='end threshold for color acc', default=sqrt(3)*255))
    parser.add_argument('-ctss', '--color_thresh_step_size', type = float, help='step size for loc acc', default=10)

    args = parser.parse_args()
    batch_size = args.batchsize
    model_path = args.model_path

    # thresholds
    location_thresh_start = args.location_thresh_start
    location_thresh_end = args.location_thresh_end
    location_thresh_step_size = args.location_thresh_step_size

    color_thresh_start = args.color_thresh_start
    color_thresh_end = args.color_thresh_end
    color_thresh_step_size = args.color_thresh_step_size

    loc_thresh = torch.linspace(location_thresh_start, location_thresh_end, steps = location_thresh_step_size)
    color_thresh = torch.linspace(color_thresh_start, color_thresh_end, steps=color_thresh_step_size)
    
    test_ds = EnvironmentJPGDataset(os.path.join('data', 'test_feature_matrix.npy'), os.path.join('data', 'test_label.npy'),\
        transform= transforms.Compose([
                                       transformer.ToTensor(),
                                       transformer.CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                       , augmentation=augmentation_param)
    test_dataloader = DataLoader(test_ds, batch_size)

    loc_thresh_vs_acc, color_thresh_vs_acc = evaluate(model, loc_thresh, color_thresh, test_dataloader)

    plot_threshold_and_accuracy(loc_thresh_vs_acc, color_thresh_vs_acc)
