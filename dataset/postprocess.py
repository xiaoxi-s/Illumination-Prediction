import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils

from scipy.interpolate import interp2d
from functools import cmp_to_key

from imutils import contours
from skimage import measure

import utils

class result_interpreter():
    def post_processing_labelsH(self, labels):
        for label in labels:
            print("rotation y:")
            print(label[1]/math.pi * 180) 
            print(" rotation z:")
            print(-label[0]/math.pi * 180 + 180)
            print(label[4], label[3], label[2])

    def post_processing_labelsL(self, labels):
        for label in labels:
            points_x = 10 * np.cos(label[0]) * np.sin(label[1])
            points_y = 10 * np.sin(label[0]) * np.sin(label[1])
            points_z = 10 * np.cos(label[1])
            print(points_x, points_y, points_z)
            print(label[4], label[3], label[2])
            print("\n")

    def EncodeToSRGB(self, v):
        if (v <= 0.0031308):
            return (v * 12.92) * 255.0
        else:
            return (1.055*(v**(1.0/2.4))-0.055) * 255.0
