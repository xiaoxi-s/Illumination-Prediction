import os
import argparse

from utils import convert_exr_and_write


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert exr to exr or jpg both of size (400, 900)')
    parser.add_argument('-f', '--imgformat', type=str,
                    help='The type images would be converted to', default='jpg')
    parser.add_argument('-dp', '--datapath', type=str, help='data source path', default='/home/adrian/Downloads/laval-dataset')
    parser.add_argument('-s', '--size', type=int, 
                    help='numebr of images converted, if is -1, convert all the image', default=30)
    args = parser.parse_args()

    data_path = args.datapath
    img_format = args.imgformat
    size = args.size

    # format check
    if str.lower(img_format) != 'jpg' and str.lower(img_format) != 'jpeg' and str.lower(img_format) != 'exr':
        raise TypeError("Format supported: jpg and exr")

    if str.lower(img_format) == 'jpeg':
        img_format= 'jpg'
    img_format = str.lower(img_format)

    # store images in different dirs based on format
    dest_path = os.path.join('data', img_format + '_sample')
    if os.path.isdir(dest_path):
        os.mkdir(dest_path)

    file_ind = 1
    exr_lists = os.listdir(data_path)
    # number of images is size 
    if size != -1 and size < len(exr_lists):
        exr_lists = exr_lists[0:size]

    # convert and write
    for exr_file_name in exr_lists:
        converted_file = os.path.join(dest_path, str(file_ind) + '.' + img_format)
        convert_exr_and_write(os.path.join(data_path, exr_file_name), converted_file, img_format)
        file_ind = file_ind + 1

