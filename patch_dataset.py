import os
import argparse
import numpy as np

from utils import patch_data_from_npz
from utils import 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='patch all training data together')
    parser.add_argument('-fp', '--folder_path', help='path to the folder that contains npz files')
    parser.add_argument('-dp', '--dest_path', help='output path')
    parser.add_argument('-fmt', '--file_format', help='file format to be patched', default='npz')
    args = parser.parse_args()

    folder_path = args.folder_path
    dest_path = args.dest_path
    
    file_format = args.file_format

    if file_format == 'npz':
        patch_data_from_npz(folder_path, dest_path)
    elif file_format == 'npy':
        # not implemented. Do this mannually
        patch_data_from_npy(folder_path, dest_path)
    else:
        raise TypeError('Tpye must be npz or npy')