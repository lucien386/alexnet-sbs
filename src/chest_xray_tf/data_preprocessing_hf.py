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
normal_test = "/Users/frank/Documents/GitHub/alexnet-sbs/dataSet/test/NORMAL"
illed_test = "/Users/frank/Documents/GitHub/alexnet-sbs/dataSet/test/PNEUMONIA"

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
label_list_normal_test, img_list_normal_test = get_image_data(normal_test)
label_list_illed_test, img_list_illed_test = get_image_data(illed_test)

print("resizing")
resized_normal_train = []
for img in img_list_normal_train:
    resized_normal_train.append(resize_img(img, shape = (227, 227)))

resized_illed_train = []
for img in img_list_illed_train:
    resized_illed_train.append(resize_img(img, shape = (227, 227)))

resized_normal_test = []
for img in img_list_normal_test:
    resized_normal_test.append(resize_img(img, shape = (227, 227)))

resized_illed_test = []
for img in img_list_illed_test:
    resized_illed_test.append(resize_img(img, shape = (227, 227)))

print("zipping")
normal_train = np.asarray(list(zip(label_list_normal_train, resized_normal_train)))
illed_train = np.asarray(list(zip(label_list_illed_train, resized_illed_train)))

normal_test = np.asarray(list(zip(label_list_normal_test, resized_normal_test)))
illed_test = np.asarray(list(zip(label_list_illed_test, resized_illed_test)))

combined_train = np.concatenate((normal_train, illed_train))
combined_test = np.concatenate((normal_test, illed_test))
np.random.shuffle(combined_train)
np.random.shuffle(combined_test)

div_and_save(combined_train, "/Users/frank/Documents/GitHub/alexnet-sbs/dataSet/parsed_train")

div_and_save(combined_test, "/Users/frank/Documents/GitHub/alexnet-sbs/dataSet/parsed_test")
