import os
import cv2
import json
import argparse
import numpy as np

from dataset.preprocess import DataGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Feature Matrix')
    parser.add_argument('-f', '--imgformat', type=str,
                    help='generate annotated images of a specific type: jpg or exr', default='exr')
    parser.add_argument('-sp', '--sourcepath', type=str, help='data source path', default=os.path.join('data', 'jpg_sample'))
    parser.add_argument('-dp', '--destpath', type=str, help='data destination path', default='data')
    parser.add_argument('-s', '--start',type=int, help='start index', default=0)
    parser.add_argument('-e', '--end', type=int, help='end index', default = -1)
    parser.add_argument('-rh', '--resize_height', type=int, help='resize height', default=1024)
    parser.add_argument('-rw', '--resize_width', type=int, help='resize width', default=2048)
    parser.add_argument('-ow', '--output_width', type=int, help='output width', default=360)
    parser.add_argument('-oh', '--output_height', type=int, help='output height', default=240)

    # format: jpg or exr
    args = parser.parse_args()
    img_format = args.imgformat
    img_format = str.lower(img_format)

    start = args.start
    end = args.end
    if img_format == 'jpg' or img_format == 'jpeg' or img_format == 'exr':
        pass
    else:
        raise TypeError("Supported format: jpg and exr")
    
    resize_height = args.resize_height
    resize_width = args.resize_width
    output_width = args.output_width
    output_height = args.output_height

    # paths
    source_path = args.sourcepath
    dest_path = args.destpath

    # generate feature matrix
    matrix_generator = DataGenerator()
    matrix_generator.generate_feature_and_label_new(source_path, dest_path, False, \
        resize_height=resize_height, resize_width=resize_width, output_width=output_width, \
        output_height=output_height,start=start, end=end)