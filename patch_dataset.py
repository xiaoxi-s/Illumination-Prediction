import os
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='patch all training data together')
    parser.add_argument('-fp', '--folder_path', help='path to the folder that contains npz files')
    parser.add_argument('-dp', '--dest_path', help='output path')
    args = parser.parse_args()

    folder_path = args.folder_path
    dest_path = args.dest_path

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
    