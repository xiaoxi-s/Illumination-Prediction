import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
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

        np.save(os.path.join(dest_path, 'labeled_images.npy'), labeled_images, allow_pickle=True)
        np.save(os.path.join(dest_path, 'labels.npy'), labels, allow_pickle=True)

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

