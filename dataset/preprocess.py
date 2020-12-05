import os
import cv2
import math
import numpy as np
from random import seed
from random import random
from pyquaternion import Quaternion

from functools import cmp_to_key
import imutils
from imutils import contours
from skimage import measure

from dataset.ImageCropper import imagecropper as ImgCp

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
                print([x,y])
                points_x = 10 * np.cos(x/img.shape[1]*math.pi*2) * np.sin(y/img.shape[0]*math.pi)
                points_y = 10 * np.sin(x/img.shape[1]*math.pi*2) * np.sin(y/img.shape[0]*math.pi)
                points_z = 10 * np.cos(y/img.shape[0]*math.pi)
                print([points_x,points_y,points_z, 1 , rect[1][0], rect[1][1], rect[2], r, g, b])

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
    def _get_threshold_img(self, img, threshold):
        """
        Return a mask of pixels' value that higher than the extreme in the image * threshold (0 < Threshold < 1)
        """
        img_one_channel = np.sum(img, axis=-1)

        # average the 20 extremes, prevent outliers
        img_sorted_one_channel = img_one_channel.flatten()
        img_sorted_one_channel = np.sort(img_sorted_one_channel)
        extremes = img_sorted_one_channel[-20:]
        extreme = extremes.mean()

        thresh_img = np.zeros(shape=img_one_channel.shape, dtype='uint8')
        thresh_img[img_one_channel >  extreme * threshold] = 255

        thresh_img = cv2.erode(thresh_img, None, iterations=2)
        thresh_img = cv2.dilate(thresh_img, None, iterations=4)

        return thresh_img

    def _find_connected_components(self, img, thresh_img, protion_of_pixel_required):
        """
        Find connected componenets in the mask. The pixel count for each componenet should exceed total_image_pixel/protion_of_pixel_required
        In short, as protion goes up, there will be more components
        """
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
            # we have requirements for low threshold image, but not for high threshold image
            if (protion_of_pixel_required == -1):
                if numPixels > 300:
                    masks.append([np.resize(labelMask, thresh_img.shape), np.sum(img[labelMask == 255])/np.count_nonzero(img[labelMask == 255]), numPixels])
            else:
                if numPixels > img.shape[0]*img.shape[1]/protion_of_pixel_required:
                    masks.append([np.resize(labelMask, thresh_img.shape), np.sum(img[labelMask == 255])/np.count_nonzero(img[labelMask == 255]), numPixels])
        
        return masks

    def _label_high(self, masks, img, N = 3):
        labels = []
        # if there is no enough bright light, add lights that have no color
        if (len(masks) < N):
            for i in range(N-len(masks)):
                labels.append([0, 0, 0, 0, 0])
        else:
            masks = masks[0:N]

        # for each illumination (denoted by mask)
        for mask, _, _ in masks:
            # find the contour
            # cv2.imshow('mask', mask)
            # cv2.waitKey(0)
            lights_position = np.where(mask == 255) 
            direction = self._average_in_spherical(lights_position, img.shape)
            color = self._find_color(img, mask)
            labels.append([direction[0], direction[1], color[0], color[1], color[2]])
        return labels


    def _label_low(self, masks, img, N = 3):
        labels = []
        # if there is no enough bright light, add lights that have no color
        if (len(masks) < N):
            for i in range(N-len(masks)):
                labels.append([0, 0, 0, 0, 0])
        else:
            masks = masks[0:N]

        for mask, _, _ in masks:
            # cv2.imshow('mask', mask)
            # cv2.waitKey(0)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = contours.sort_contours(cnts)[0]

            # loop over the contours
            for (i, c) in enumerate(cnts):
                # find a rectangle (ellipse)
                rect = cv2.minAreaRect(c)
                x, y = rect[0]
                theta = x/img.shape[1] * 2 * math.pi
                phi = y/img.shape[0] * math.pi
                color = self._find_color(img, mask)
                labels.append([theta, phi, color[0], color[1], color[2]])
                
        return labels
            
    def _find_color(self, img, mask):
        """
        Find the average color of that region
        """
        image_size = img.shape
        element = np.count_nonzero(mask)
        counting = np.zeros(img.shape)
        counting[mask == 255] = img[mask == 255]
        blue = np.sum(counting[:,:,0]) / element 
        green = np.sum(counting[:,:,1]) / element
        red = np.sum(counting[:,:,2]) / element
        return [blue, green, red]

    def _average_in_spherical(self, points, img_size):
        """
        Find the average position of those points in spherical coordinate
        """
        points_phi = points[0]
        points_theta = points[1]

        points_phi = points_phi / img_size[0] * math.pi
        points_theta = points_theta / img_size[1] * 2 * math.pi

        points_x = 1 * np.cos(points_theta) * np.sin(points_phi)
        points_y = 1 * np.sin(points_theta) * np.sin(points_phi)
        points_z = 1 * np.cos(points_phi)

        average_points_x = np.mean(points_x)
        average_points_y = np.mean(points_y)
        average_points_z = np.mean(points_z)

        theta = np.arctan2(average_points_y, average_points_x)
        if theta < 0:
            theta = theta + math.pi*2

        phi = np.arccos(average_points_z/np.linalg.norm([average_points_x,average_points_y,average_points_z]))
        return [theta, phi]

    def generate_labels(self, img, N=3):
        # preprocessing
        H_threshold = 0.1
        L_threshold = 0.01
        L_portion_of_pixels_required = 500
        maskH = []
        maskL = []
        iteration = 0
        # adjust the threshold to make sure there is 3 bright lights and 3 indirect lights 
        while(iteration < 10):
            iteration += 1
            thresh_img_high = self._get_threshold_img(img, H_threshold)
            thresh_img_low = self._get_threshold_img(img, L_threshold)
            # cv2.imshow('mask', thresh_img_high)
            # cv2.waitKey(0)
            # cv2.imshow('mask', thresh_img_low)
            # cv2.waitKey(0)
            blur = cv2.GaussianBlur(thresh_img_high,(11,11),0)
            thresh_img_low[blur > 0] = 0
        
            maskH = self._find_connected_components(img, thresh_img_high, -1)
            maskL = self._find_connected_components(img, thresh_img_low, L_portion_of_pixels_required)

            # print(len(maskH), len(maskL))
            # print(H_threshold, L_threshold, L_portion_of_pixels_required)

            if (len(maskH) < N and H_threshold > 0.01):
                H_threshold -= 0.01 / math.sqrt(iteration) * abs(len(maskH) - N)        
            elif (len(maskH) > N and H_threshold < 0.5):
                H_threshold += 0.01 / math.sqrt(iteration) * abs(len(maskH) - N) 
            
            if (len(maskL) < N and L_threshold > 0.005):
                L_threshold -= 0.001 / math.sqrt(iteration) * abs(len(maskL) - N)   
                L_portion_of_pixels_required += 50 / iteration * abs(len(maskL) - N) 
            elif (len(maskL) > N and L_threshold < H_threshold/4):
                L_threshold += 0.001 / math.sqrt(iteration) * abs(len(maskL) - N) 
                L_portion_of_pixels_required -= 50 / iteration * abs(len(maskL) - N) 
            
            if(len(maskL) == N and len(maskH) == N):
                break   
        
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

        # print(len(maskH), len(maskL))
        # print(H_threshold, L_threshold, L_portion_of_pixels_required)

        if (len(maskH) == 0):
            return None

        # otherwise, return the first N largest light
        maskH = sorted(maskH, key =cmp_to_key(compare_connected_components), reverse=True)
        maskL = sorted(maskL, key =cmp_to_key(compare_connected_components), reverse=True)
        # print(len(maskH), len(maskL))
        labelH = self._label_high(maskH, img)
        labelL = self._label_low(maskL, img)
        
        return labelH+labelL

class DataGenerator():
    def __init__(self):
        pass
    def translate_cropped_image(self, new_theta, new_phi, labels):
        """
        translate all labels using view vector as unit vector
        """
        finallabel = []
    
        for label in labels:
            theta = label[0] - new_theta
              
            phi = label[1]         
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            up = Quaternion(axis=[0, 1, 0], angle=math.pi / 2 - new_phi)
            v = np.array([x, y, z]) 
            v_prime = up.rotate(v)
            theta = np.arctan2(v_prime[1], v_prime[0])
            phi = np.arccos(v_prime[2]/np.linalg.norm(v_prime))
            if theta < 0:
                theta += math.pi * 2 
            finallabel.append(np.array([theta, phi, label[2], label[3], label[4]]).astype(dtype = 'float64'))
            
        return finallabel


    def generate_feature_and_label_new(self, source_path, dest_path, show_image):
        """
        Set show_image to true to view each images, and its view vector
        """
        seed(10)
        labeler = EXRLabeler()
        
        img_names = os.listdir(os.path.join(source_path))
        img_names.sort()

        discarded_data_set = set()
        label_dict = dict()
        
        img_index = 0
        
        labeled_images = []
        labels = []

        # generate labels
        for img_name in img_names:

            if img_index % 10 == 0:
                print("generating labels for image ", img_index+1, " to ", img_index+10)

            img = cv2.imread(os.path.join(source_path, img_name), cv2.IMREAD_UNCHANGED)
            label = labeler.generate_labels(img)
            cropper = ImgCp(img)

            if label is None:
                discarded_data_set.add(img_name)
                continue
            
            # crop 4 images in 4 random locations
            for i in range(4):
                theta = random() * 2 * math.pi
                phi = 2 * math.pi / 5 + (random() * math.pi/5)

                camera_img = cropper.generate_image(theta, phi, math.pi/2, 720, 480)
                if show_image:
                    print("viewing theta", theta, "viewing phi", phi)
                    im = cv2.normalize(camera_img, None, alpha=0, beta = 500, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imshow('each_image', im)
                    cv2.waitKey(0)
                
                transformed_labels = self.translate_cropped_image(theta, phi, label)

                labels.append(transformed_labels)
                labeled_images.append(camera_img)

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


# test
# if __name__ == '__main__':
#     # preprocessing
#     img = cv2.imread('./test_conversion/5.jpg')
#     l = ImageLabeler()
#     thresh_img = l._get_threshold_img(img)
#     masks = l._find_connected_components(img, thresh_img, 3)
#     img = l._annotate_image(masks, img)

#     for mask in masks:
#         plt.imshow(mask)
#         plt.show()

    # exr => label, jpeg => train
    # ambient light
    # chopped image

