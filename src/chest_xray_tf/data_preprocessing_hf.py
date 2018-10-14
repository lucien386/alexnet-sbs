# Functions that preprocess (NOT FETCH) data should be put in here
import numpy as np
import tensorflow as tf
from fetch_hf import get_image_data
import os
import matplotlib.pyplot as plt
import cv2

sess = tf.InteractiveSession()
normal_train = "/Users/frank/Documents/GitHub/alexnet-sbs/dataSet/train/NORMAL"
illed_train = "/Users/frank/Documents/GitHub/alexnet-sbs/dataSet/train/PNEUMONIA"

def largest_image_size(img_array): #Tom: possible np.amax() faster execution
    max_row = float('-inf')
    max_column = float('-inf')
    for i in range(1, len(img_array), 2):
        array = img_array[i]
        if len(array) > max_row:
            max_row = len(array)
        if len(array[0]) > max_column:
            max_column = len(array[0])
    largest_size = (max_row, max_column)
    return largest_size


def smallest_image_size(img_array): #Tom: possible np.amax() faster execution
    min_row = float('inf')
    min_column = float('inf')
    for i in range(1, len(img_array), 2):
        array = img_array[i]
        if len(array) < min_row:
            min_row = len(array)
        if len(array[0]) < min_column:
            min_column = len(array)
    largest_size = (min_row, min_column)
    return largest_size


def resize_img(img, shape):
    resized = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    return resized


def div_and_save(lis, path, batch_size = 100):
    max_len = len(lis)
    for i in range(0, len(lis), batch_size):
        temp = lis[i:min(i+100, max_len)]
        file = path + "/" + "batch" + str(int(i/batch_size)) + ".npy"
        print(file)
        np.save(file, temp)


print("loading")
label_list_normal_train, img_list_normal_train = get_image_data(normal_train)
label_list_illed_train, img_list_illed_train = get_image_data(illed_train)

print("resizing")
resized_normal = []
for img in img_list_normal_train:
    resized_normal.append(resize_img(img, shape = (227, 227)))

resized_illed = []
for img in img_list_illed_train:
    resized_illed.append(resize_img(img, shape = (227, 227)))

print("zipping")
normal = np.asarray(list(zip(label_list_normal_train, resized_normal)))
illed = np.asarray(list(zip(label_list_illed_train, resized_illed)))

combined = np.concatenate((normal, illed))
np.random.shuffle(combined)
div_and_save(combined, "/Users/frank/Documents/GitHub/alexnet-sbs/dataSet/parsed_train")
