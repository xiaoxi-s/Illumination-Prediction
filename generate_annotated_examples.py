import os
import cv2
import argparse
import numpy as np

from dataset.preprocess import JPGLabeler, EXRLabeler 


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Generate example images')
    parser.add_argument('-f', '--imgformat', type=str,
                    help='generate annotated images of a specific type: jpg or exr', default='exr')
    parser.add_argument('-dp', '--datapath', type=str, help='data source path', default='exr_sample')
    args = parser.parse_args()

    img_format = args.imgformat

    # decide a labeler
    if str.lower(img_format) == 'jpg' or str.lower(img_format) == 'jpeg':
        labler = JPGLabeler()
    elif str.lower(img_format) == 'exr':
        labler = EXRLabeler()
    else:
        raise TypeError("Format supported: jpg and exr")

    data_path = args.datapath
    print(data_path)
    example_path = os.path.join('data', 'annotated')
    print(example_path)
    if not os.path.isdir(example_path):
        os.mkdir(example_path)

    # get image files
    img_names = os.listdir(os.path.join(data_path))
    discarded_data_set = set()
    img_names.sort()

    # generate images
    for img_name in img_names:
        img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_UNCHANGED).astype(np.float32)
        img = labler.generate_annotated_img(img)
        # any img with fewer than 3 lights is removed. 
        if img is None:
            discarded_data_set.add(img_name)
            continue

        # write images to the specific location
        cv2.imwrite(os.path.join(example_path, img_name),img)

    
