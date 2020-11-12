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


# extract max

# grow the seed

# find illuminated part

# repeat


class JPGLabeler():
    def _get_threshold_img(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (11, 11), 0)
        extreme = blurred_img.max()

        img_mean = np.mean(img)

        if img_mean < 85:
            thresh_img = cv2.threshold(blurred_img, extreme*5/12, extreme, cv2.THRESH_BINARY)[1]
        else:
            thresh_img = cv2.threshold(blurred_img, extreme*2/3, extreme, cv2.THRESH_BINARY)[1]
        
        thresh_img = cv2.erode(thresh_img, None, iterations=2)
        thresh_img = cv2.dilate(thresh_img, None, iterations=4)

        return thresh_img


    def _find_connected_components(self, img, thresh_img, N):
        labels = measure.label(thresh_img, connectivity=1, background=0)
        masks = []
        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue

            # otherwise, construct the label mask and count the
            # number of pixels 
            labelMask = np.zeros(thresh_img.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            if numPixels > 300:
                masks.append([np.resize(labelMask, thresh_img.shape), np.sum(img[labelMask == 255])/np.count_nonzero(img[labelMask == 255]), numPixels])
        
        # If not enough number of light, discard the image
        if len(masks) < N:
            return None
        
        def compare_connected_components(x, y):
            if x[1] > y[1]:
                return 1
            elif x[1] < y[1]:
                return -1
            elif x[2] > y[2]:
                return 1
            elif x[2] < y[2]:
                return -1
            else:
                return 0

        # otherwise, return the first N largest light
        masks = sorted(masks, key =cmp_to_key(compare_connected_components), reverse=True)
        return masks[0:N]

    
    def _label_img(self, masks, img):
        # for l in labels: 
        # l = [[(x1, y1, 1), (ax1, ax2), rotation1, ('R', 'G', 'B')], 
        #      [(x2, y2, 1), (ax1, ax2), rotation2, ('R', 'G', 'B')],
        #      ...,
        #      [(xN, yN, 1), (ax1, ax2), rotationN, ('R', 'G', 'B')]]

        labels = []
        # for each illumination (denoted by mask)
        for mask, _, _ in masks:
            # find the contour
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = contours.sort_contours(cnts)[0]
            
            # loop over the contours
            for (i, c) in enumerate(cnts):
                
                # draw the bright spot on the image
                rect = cv2.minAreaRect(c)
                x, y = rect[0]
                
                r, g, b =np.divide(np.sum(img[mask == 255], axis=(0)), \
                    np.count_nonzero(img[mask==255], axis=(0)))

                labels.append([x, y, 1 , rect[1][0], rect[1][1], rect[2], r, g, b])

        return labels


    def _annotate_img(self, masks, img):

        for mask, _, _ in masks:
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = contours.sort_contours(cnts)[0]

            # loop over the contours
            for (i, c) in enumerate(cnts):
                # draw the bright spot on the image
                rect = cv2.minAreaRect(c)
                elp = cv2.ellipse(img, rect, color = (0, 0, 255), thickness=2)

        return img


    def generate_annotated_img(self, img, N=3):
        # preprocessing
        thresh_img = self._get_threshold_img(img)
        masks = self._find_connected_components(img, thresh_img, N)
        
        if masks is None:
            return None
        
        img = self._annotate_img(masks, img)

        return img


    def generate_labels(self, img, N=3):
        # preprocessing
        thresh_img = self._get_threshold_img(img)
        masks = self._find_connected_components(img, thresh_img, N)
        if masks is None:
            return None
        
        label = self._label_img(masks, img)
        return label


class EXRLabeler():
    def _get_threshold_img(self, img):
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blurred_img = cv2.GaussianBlur(gray_img, (11, 11), 0)
        img_one_channel = np.sum(img, axis=-1)
        extreme = img_one_channel.max()
        

        # img_mean = np.mean(img)

        # if img_mean < 85:
        #     thresh_img = cv2.threshold(blurred_img, extreme*5/12, extreme, cv2.THRESH_BINARY)[1]
        # else:
        #     thresh_img = cv2.threshold(blurred_img, extreme*2/3, extreme, cv2.THRESH_BINARY)[1]
        
        # thresh_img = cv2.erode(thresh_img, None, iterations=2)
        # thresh_img = cv2.dilate(thresh_img, None, iterations=4)
        thresh_img = np.zeros(shape=img_one_channel.shape, dtype='uint8')
        thresh_img[img_one_channel >  extreme*1/100] = 255

        return thresh_img


    def _find_connected_components(self, img, thresh_img, N):
        labels = measure.label(thresh_img, connectivity=1, background=0)
        masks = []
        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue

            # otherwise, construct the label mask and count the
            # number of pixels 
            labelMask = np.zeros(thresh_img.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            if numPixels > 300:
                masks.append([np.resize(labelMask, thresh_img.shape), np.sum(img[labelMask == 255])/np.count_nonzero(img[labelMask == 255]), numPixels])
        
        # If not enough number of light, discard the image
        if len(masks) < N:
            return None
        
        def compare_connected_components(x, y):
            if x[1] > y[1]:
                return 1
            elif x[1] < y[1]:
                return -1
            elif x[2] > y[2]:
                return 1
            elif x[2] < y[2]:
                return -1
            else:
                return 0

        # otherwise, return the first N largest light
        masks = sorted(masks, key =cmp_to_key(compare_connected_components), reverse=True)
        return masks[0:N]

    
    def _label_img(self, masks, img):
        # for l in labels: 
        # l = [[(x1, y1, 1), (ax1, ax2), rotation1, ('R', 'G', 'B')], 
        #      [(x2, y2, 1), (ax1, ax2), rotation2, ('R', 'G', 'B')],
        #      ...,
        #      [(xN, yN, 1), (ax1, ax2), rotationN, ('R', 'G', 'B')]]

        labels = []
        
        # for each illumination (denoted by mask)
        for mask, _, _ in masks:
            # find the contour
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = contours.sort_contours(cnts)[0]

            # loop over the contours
            for (i, c) in enumerate(cnts):
                # find a rectangle (ellipse)
                rect = cv2.minAreaRect(c)
                x, y = rect[0]
                
                r, g, b =np.divide(np.sum(img[mask == 255], axis=(0)), \
                    np.count_nonzero(img[mask==255], axis=(0)))

                labels.append([x, y, 1 , rect[1][0], rect[1][1], rect[2], r, g, b])

        return labels


    def _annotate_img(self, masks, img):
        max_intensity = np.max(img)
        for mask, _, _ in masks:
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = contours.sort_contours(cnts)[0]

            # loop over the contours
            for (i, c) in enumerate(cnts):
                # draw the bright spot on the image
                rect = cv2.minAreaRect(c)
                elp = cv2.ellipse(img, rect, color = (0, 0, 255), thickness=2)

        return img


    def generate_annotated_img(self, img, N=3):
        # preprocessing
        thresh_img = self._get_threshold_img(img)
        masks = self._find_connected_components(img, thresh_img, N)
        
        if masks is None:
            return None
        
        img = self._annotate_img(masks, img)

        return img


    def generate_labels(self, img, N=3):
        # preprocessing
        thresh_img = self._get_threshold_img(img)
        masks = self._find_connected_components(img, thresh_img, N)
        if masks is None:
            return None
        
        label = self._label_img(masks, img)
        return label


class DataGenerator():
    def __init__(self):
        pass

    def generate_feature_and_label(self, source_path, dest_path, img_format='jpg'):
        img_format = str.lower(img_format)
        if img_format == 'jpg' or img_format == 'jpeg':
            labeler = JPGLabeler()
        elif img_format == 'exr':
            labeler = EXRLabeler()
        
        img_names = os.listdir(os.path.join(source_path))
        discarded_data_set = set()
        img_names.sort()
        label_dict = dict()
        
        img_index = 0
        
        labeled_images = []
        labels = []

        # generate labels
        for img_name in img_names:
            img = cv2.imread(os.path.join(source_path, img_name))
            label = labeler.generate_labels(img)

            if label is None:
                discarded_data_set.add(img_name)
                continue

            # label_dict[img_index] = label

            labels.append(label)
            labeled_images.append(img)

            img_index += 1

        # store labeled images
        # with open(os.path.join(label_path, 'labels.txt'), 'w') as f:
        #    f.write(json.dumps(label_dict))

        labeled_images = np.array(labeled_images)
        labels = np.array(labels, dtype='float64')

        # num of X == num of Y
        assert labels.shape[0] == labeled_images.shape[0]

        # ratio of test sample
        ratio = 0.2
        num_of_samples = labeled_images.shape[0]
        num_of_test_samples = int(ratio * num_of_samples)
        num_of_train_samples = int(num_of_samples - num_of_test_samples)

        train_matrix = labeled_images[0:num_of_train_samples]
        train_labels = labels[0:num_of_train_samples]

        test_matrix = labeled_images[num_of_train_samples:-1]
        test_labels = labels[num_of_train_samples:-1]

        np.save(os.path.join(dest_path, 'train_feature_matrix.npy'), train_matrix, allow_pickle=True)
        np.save(os.path.join(dest_path, 'train_label.npy'), train_labels, allow_pickle=True)

        np.save(os.path.join(dest_path, 'test_feature_matrix.npy'), test_matrix, allow_pickle=True)
        np.save(os.path.join(dest_path, 'test_label.npy'), test_labels, allow_pickle=True)

class sphericalSystem():
    def __init__(self, environment_map):
        self.map = environment_map
        self.height = environment_map.shape[0]
        self.width = environment_map.shape[1]
        self.type = environment_map.dtype
        self.channel = environment_map.shape[2]

    def GenerateImage(self, starting_theta, starting_phi, theta_size, phi_size, height_resolution, width_resolution):
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


# test
if __name__ == '__main__':
    # preprocessing
    img = cv2.imread('./test_conversion/5.jpg')
    l = ImageLabeler()
    thresh_img = l._get_threshold_img(img)
    masks = l._find_connected_components(img, thresh_img, 3)
    img = l._annotate_image(masks, img)

    for mask in masks:
        plt.imshow(mask)
        plt.show()

    # exr => label, jpeg => train
    # ambient light
    # chopped image

