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

class imagecropper():
    """
    This class takes in an environment map and crop image at different orientations
    """
    def __init__(self, environment_map):
        self.map = environment_map
        self.height = environment_map.shape[0]
        self.width = environment_map.shape[1]
        self.type = environment_map.dtype
        self.channel = environment_map.shape[2]

    def convert_to_spherical_phi(self, index_row, image_height):
        return index_row/image_height * math.pi

    def convert_to_spherical_theta(self, index_col, image_width):
        return index_col/image_width * 2* math.pi

    def generate_image(self, view_direction_theta, view_direction_phi, FOV_horizontal, image_width, image_height):
        """
        Return a cropped image

        Parameters
        ----------
        view_direction_theta : float.
            The theta of the view vector, which defines in spherical coordinate

        view_direction_phi : float.
            The phi of the view vector, which defines in spherical coordinate

        FOV_horizontal : float.
            The field of view of the cropped image horizontally

        image_height, image_width : int.

        Returns
        -------
        out : 2darray
            A 2darray with size = image_heigh x image_width

        Examples
        --------
        image = handle.generate_image(0, 0, math.pi/3, 1080, 720)

        """
        FOV_vertical = FOV_horizontal/image_width * image_height
        theta_start = view_direction_theta - FOV_horizontal/2
        phi_start = view_direction_phi - FOV_vertical/2

        image_center = np.array([1 * np.cos(view_direction_theta) * np.sin(view_direction_phi), \
            1 * np.sin(view_direction_theta) * np.sin(view_direction_phi),1 * np.cos(view_direction_phi)])
        horizontal_size_half = np.tan(FOV_horizontal/2)
        vertical_size_half = np.tan(FOV_vertical/2)
        
        horizontal_displacement = np.array([-horizontal_size_half * np.sin(view_direction_theta), horizontal_size_half * np.cos(view_direction_theta), 0])
        vertical_displacement = np.array([-vertical_size_half*np.cos(view_direction_phi)*np.cos(view_direction_theta), -vertical_size_half * np.cos(view_direction_phi) * np.sin(view_direction_theta), \
            vertical_size_half*np.sin(view_direction_phi)])

        top_left = image_center + vertical_displacement - horizontal_displacement
        top_right = image_center + vertical_displacement + horizontal_displacement
        bot_left = image_center - vertical_displacement - horizontal_displacement
        bot_right = image_center - vertical_displacement + horizontal_displacement

        # form the image plane
        
        img_rect = np.zeros(shape=(image_height,image_width, 3), dtype=np.float32)
        left = top_left
        right = top_right
        height_array = np.linspace(top_left ,bot_left, image_height)
        for i in range(image_height):
            left = height_array[i]
            right = left + horizontal_displacement*2
            img_rect[i,:,:] = np.linspace(left,right,image_width)
        # transfer the image plane back to spherical 
        theta = np.arctan2(img_rect[:,:,1], img_rect[:,:,0])
        theta[theta < 0] = theta[theta < 0] + math.pi*2
        phi = np.arccos(img_rect[:,:,2]/np.linalg.norm(img_rect, axis=2))

        # Read from the original image
        # u = np.floor(theta  * self.width / math.pi)
        # v = np.floor(phi * self.height / (math.pi/2))
        
        # img = np.zeros(shape=(image_height,image_width, self.channel), dtype=self.type)
        # for i in range(self.channel):
        #     f = interp2d(np.linspace(0, math.pi * 2,self.width), np.linspace(0,math.pi, self.height), self.map[:,:,i], kind='linear')
        #     for j in range(theta.shape[0]):
        #         for k in range(theta.shape[1]):
        #             img[j,k,i]  = (f(theta[j,k], phi[j,k]))[0]

        img2 = np.zeros(shape=(image_height,image_width, self.channel), dtype=self.type)

        theta = theta / 2 / math.pi * self.width
        phi = phi / math.pi * self.height
        theta = np.round(theta).astype('int') 
        phi = np.round(phi).astype('int')
 
        theta[theta >= self.width] = self.width-1
        theta[theta < 0] = 0
        phi[phi >= self.height] = self.height-1
        phi[phi < 0] = 0
        for i in range(self.channel):
            img2[:,:,i] = self.map[phi, theta,i]
        
        return cv2.GaussianBlur(img2,(3,3),0)

    def generate_image_detail(self, starting_theta, starting_phi, theta_size, phi_size, height_resolution, width_resolution):
        # create four points on the sphere
        top_left_sph = np.array([1, starting_theta, starting_phi], dtype = np.float32)
        top_right_sph = np.array([1, starting_theta + theta_size, starting_phi], dtype = np.float32)
        bot_left_sph = np.array([1, starting_theta, starting_phi+phi_size], dtype = np.float32)
        bot_right_sph = np.array([1, starting_theta + theta_size, starting_phi+phi_size], dtype = np.float32)

        # transfer them to rect system
        top_left = np.array([top_left_sph[0] * np.cos(top_left_sph[1]) * np.sin(top_left_sph[2]), \
            top_left_sph[0] * np.sin(top_left_sph[1]) * np.sin(top_left_sph[2]), top_left_sph[0] * np.cos(top_left_sph[2])])
        top_right = np.array([top_right_sph[0] * np.cos(top_right_sph[1]) * np.sin(top_right_sph[2]), \
            top_right_sph[0] * np.sin(top_right_sph[1]) * np.sin(top_right_sph[2]), top_right_sph[0] * np.cos(top_right_sph[2])])
        bot_left = np.array([bot_left_sph[0] * np.cos(bot_left_sph[1]) * np.sin(bot_left_sph[2]), \
            bot_left_sph[0] * np.sin(bot_left_sph[1]) * np.sin(bot_left_sph[2]), bot_left_sph[0] * np.cos(bot_left_sph[2])])
        bot_right = np.array([bot_right_sph[0] * np.cos(bot_right_sph[1]) * np.sin(bot_right_sph[2]), \
            bot_right_sph[0] * np.sin(bot_right_sph[1]) * np.sin(bot_right_sph[2]), bot_right_sph[0] * np.cos(bot_right_sph[2])])

        # form the image plane
        img_width_difference = top_right - top_left
        img_height_difference = bot_left - top_left
        img_rect = np.zeros(shape=(height_resolution,width_resolution, 3), dtype=np.float32)

        left = top_left
        right = top_right
        height_array = np.linspace(top_left ,bot_left, height_resolution)
        for i in range(height_resolution):
            left = height_array[i]
            right = left + img_width_difference
            img_rect[i,:,:] = np.linspace(left,right,width_resolution)

        # transfer the image plane back to spherical 
        theta = np.arctan2(img_rect[:,:,1], img_rect[:,:,0])
        theta[theta < 0] = theta[theta < 0] + math.pi/2
        phi = np.arccos(img_rect[:,:,2]/np.linalg.norm(img_rect, axis=2))

        # print(theta)
        # print(phi)

        # Read from the original image
        u = np.floor(theta  * self.width / math.pi)
        v = np.floor(phi * self.height / (math.pi/2))

        img = np.zeros(shape=(height_resolution,width_resolution, self.channel), dtype=self.type)

        # x = np.linspace(0, math.pi,self.width)
        # y = np.linspace(0, math.pi/2,self.height)
        # points_x, points_y = np.meshgrid(x, y)
        # points_x_lin = np.reshape(points_x, -1)
        # points_y_lin = np.reshape(points_y, -1)
        
        # points = np.stack((points_x_lin, points_y_lin), axis = 1)
        # print(points.shape)

        # grid_x, grid_y = np.meshgrid(u,v)
        # print(u.shape)   
        for i in range(self.channel):
            f = interp2d(np.linspace(0, math.pi * 2,self.width), np.linspace(0,math.pi, self.height), self.map[:,:,i], kind='linear')
            for j in range(theta.shape[0]):
                img[j,:,i]  = (f(theta[j,:], phi[j,:]))[0,:]
            # print(img[:,:,i])
        # for i in range(self.channel):
        #     img[:,:,i] = self.map[v,u,i]
        #     print(img[:,:,i]) 
        return img
