import os
import numpy as np
import cv2
from numpy import newaxis
# Functions fetching unprocessed data should be here


# Given the address of a folder, parse all the .jpeg files within the folder.
# Return: an numpy array consisting 1. the labels of the images
#                                   2. the numpy array version of the images
def get_image_data(folder_address):
    label_list = []
    image_list = []
    directory = os.fsencode(folder_address)
    for file in os.listdir(directory):  # Iterate through the folder
        filename = os.fsdecode(file)  # File name in string
        print("Parsing: " + filename + "...")
        if filename.endswith(".jpeg"):
            img_path = folder_address + "/" + filename
            img_array = cv2.imread(img_path, 0)  # Converting image into numpy array
            img_array = np.reshape(img_array, img_array.shape + (1,))
            if filename.find("bacteria") != -1:
                label_list.append("bacteria")
                image_list.append(img_array)
            elif filename.find("virus") != -1:
                label_list.append("virus")
                image_list.append(img_array)
            else:
                label_list.append("normal")
                image_list.append(img_array)
        print("Done.")
    image_list = np.array(image_list, dtype=object)
    label_list = np.array(label_list, dtype=object)
    return label_list, image_list
