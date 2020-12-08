import os
import torch
import argparse

import dataset.transformer as transformer
from torch.utils.data import DataLoader

from utils import load_model
from dataset.postprocess import result_interpreter
from dataset.dataset import EnvironmentJPGDataset
from torchvision import transforms

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('-mp', '--model_path', help='path to the model')
    parser.add_argument('-mn', '--model_name', help='file name of the model')
    parser.add_argument('-dp', '--dataset_path', help='dataset path')
    parser.add_argument('-destp', '--dest_path', help='prediction destination path')
    parser.add_argument('-mt', '--model_type', help='model tpye')
    parser.add_argument('-bs', '--batch_size', help='batch size', default=16)
    parser.add_argument('-N', '--N', help='num of light sources', default=6)
    parser.add_argument('-np', '--num_of_param', help='num of param of each light source', default=5)
    parser.add_argument('-sf', '--scaling_factor', type=float, help='scaling factor for exr files', default=20)
    
    args = parser.parse_args()

    model_path = args.model_path
    model_name = args.model_name
    model_type = args.model_type
    scaling_factor = args.scaling_factor

    dataset_path = args.dataset_path
    dest_path = args.dest_path

    batch_size = args.batch_size
    N = args.N
    num_of_param = args.num_of_param

    # model 
    model = load_model(os.path.join(model_path, model_name))
    device = torch.device('cuda' if torch.cuda.is_available() else ('cpu'))
    model.to(device)
    model.eval()

    # dataset & dataloader (if needed)
    test_ds = EnvironmentJPGDataset(os.path.join(dataset_path, 'test_feature_matrix.npy'), os.path.join(dataset_path, 'test_label.npy'),model_type,\
        transform= transforms.Compose([transformer.CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), scaling_factor])
                                       )

    # size of X: (num of instances, channel num, height, width)
    # size of predicted y: (num of instances, N x num_of_parm)
    X = test_ds.environment_frame
    predicted_y = torch.zeros(len(X), num_of_param)

    ind = 0
    for x in X:
        x = x.to(device)
        x = x.unsqueeze(0)
        predicted_y[ind] = model(x)[0]
        ind += 1
    
    predicted_y = predicted_y.to('cpu')

    # ----------------------------------------------
    # Missing post processing
    # -----------------------------------------------

    torch.save(predicted_y, os.join(dest_path, 'predicted_y'))
