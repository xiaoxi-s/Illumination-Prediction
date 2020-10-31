import numpy as np
import cv2
import matplotlib.pyplot as plt


def convert_exr_and_write(exr, dest_path, img_format, default_height = 400, default_width = 900):
    im=cv2.imread(exr,-1)
    height, width = im.shape[:2]

    # convert to jpg
    if img_format == 'jpg':
        tonemap = cv2.createTonemap(gamma=1)
        im = tonemap.process(im)
        im = im[0:height-600,:]

        im = cv2.normalize(im, None, alpha=0, beta=20000, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im=np.uint16(im)
        
        im = cv2.resize(im, (default_width, default_height))
        
        cv2.imwrite(dest_path, im)
    elif img_format == 'exr': # convert to exr
        tonemap = cv2.createTonemap(gamma=1)
        im = tonemap.process(im)
        im = im[0:height-600,:]

        im = cv2.resize(im, (default_width, default_height))
        
        cv2.imwrite(dest_path, im)
    else:
        raise TypeError('Supported format jpg and exr')

def imshow(img):
    plt.imshow(img)
    plt.show()