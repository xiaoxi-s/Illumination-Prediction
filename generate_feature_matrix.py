import os
import cv2
import json
import argparse
import numpy as np

from preprocess import DataGenerator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Feature Matrix')
    parser.add_argument('-f', '--imgformat', type=str,
                    help='generate annotated images of a specific type: jpg or exr', default='jpg')
    parser.add_argument('-sp', '--sourcepath', type=str, help='data source path', default=os.path.join('data', 'jpg_sample'))
    parser.add_argument('-dp', '--destpath', type=str, help='data destination path', default='data')

    # format: jpg or exr
    args = parser.parse_args()
    img_format = args.imgformat
    img_format = str.lower(img_format)
    if img_format == 'jpg' or img_format == 'jpeg' or img_format == 'exr':
        pass
    else:
        raise TypeError("Supported format: jpg and exr")
    
    # paths
    source_path = args.sourcepath
    dest_path = args.destpath

    # generate feature matrix
    matrix_generator = DataGenerator()
    matrix_generator.generate_feature_and_label(os.path.join('data', 'jpg_sample'), 'data', img_format)