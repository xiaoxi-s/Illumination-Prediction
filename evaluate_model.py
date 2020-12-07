import os
import torch
import argparse

from torch.utils.data import DataLoader

from utils import load_model
from dataset.postprocess import result_interpreter
from dataset.dataset import EnvironmentJPGDataset

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('-mp', '--model_path', help='path to the model')
    parser.add_argument('-mn', '--model_name', help='file name of the model')
    parser.add_argument('-dp', '--dataset_path', help='dataset path')
    parser.add_argument('-destp', '--dest_path', help='prediction destination path')
    parser.add_argument('-N', '--N', help='num of light sources', default=6)
    parser.add_argument('-np', '--num_of_param', help='num of param of each light source', default=5)

    args = parser.parse_args()

    model_path = args.model_path
    model_name = args.model_name

    dataset_path = args.dataset_path
    dest_path = args.dest_path

    N = args.N
    num_of_param = args.num_of_param

    # model 
    model = load_model(os.path.join(model_path, model_name))
    device = torch.device('cuda' if torch.cuda.is_available() else ('cpu'))
    model.to(device)
    model.eval()

    # dataset & dataloader (if needed)
    test_ds = EnvironmentJPGDataset(os.path.join(dataset_path, 'test_feature_matrix.npy'), os.path.join(dataset_path, 'test_label.npy'),model_type,\
        transform= transforms.Compose([transformer.CustomNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                       , augmentation=augmentation_param)
    test_dataloader = DataLoader(test_ds, batch_size, shuffle=False)

    # Or just extract feature matrix and labels
    X = test_ds.environment_frame
    y = test_ds.labels

    # size of X: (num of instances, channel num, height, width)
    # size of predicted y: (num of instances, N x num_of_parm)
    predicted_y = model(X)
    predicted_y = predicted_y.to('cpu')

    # ----------------------------------------------
    # Missing post processing
    # -----------------------------------------------

    torch.save(predicted_y, os.join(dest_path, 'predicted_y'))
