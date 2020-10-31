import os
from preprocess import ImageLabeler 
import cv2
import json
import numpy as np

class DataGenerator():
    def __init__(self):
        pass

    def generate_feature_and_label(self, source_path, dest_path):
        labeler = ImageLabeler()
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


if __name__ == '__main__':
    jpg_path = 'converted_images'
    img_names = os.listdir(os.path.join(jpg_path))
    data_path = 'data'

    
