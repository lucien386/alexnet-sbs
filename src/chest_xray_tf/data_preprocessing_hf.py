# Functions that preprocess (NOT FETCH) data should be put in here
import numpy as np
import tensorflow as tf
from fetch_hf import get_image_data
import matplotlib.pyplot as plt

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


def resize_img_numpy(img_list, shape):
    result = []
    print("Resizing images...")
    for img in img_list:
        img = tf.constant(img)
        img = tf.image.resize_images(img, shape)
        result.append(img)
    print("Done.")
    return result


normal_train = "/Users/frank/Documents/GitHub/alexnet-sbs/dataSet/train/NORMAL"
illed_train = "/Users/frank/Documents/GitHub/alexnet-sbs/dataSet/train/PNEUMONIA"
label_list_normal_train, img_list_normal_train = get_image_data(normal_train)
label_list_illed_train, img_list_illed_train = get_image_data(illed_train)

print(label_list_illed_train)

size = tf.constant([273, 273])

# img_dir = "/Users/frank/Documents/Github/alexnet-sbs/dataSet/train/normal_batch"
#
# batch_size = 100
# batch_index = 0
# sess = tf.InteractiveSession()
#
# for start in range(0, len(img_list), batch_size):
#     end = min(len(img_list), start+100)
#     curr_batch = resize_img_numpy(img_list[start:end], size)
#     curr_batch = sess.run(curr_batch)
#     print(type(curr_batch))
#     np.save(img_dir+"/batch"+str(batch_index), curr_batch)
#     batch_index += 1

# test = np.load('/Users/frank/Documents/Github/alexnet-sbs/dataSet/train/normal_batch/batch0.npy')
# im = test[0].reshape([100,100])
# print(max(im.flatten()))
# plt.imshow(im, cmap="gray")
# plt.show()
# print(1)

# test = resize_img_numpy(img_list[:800], size)
#
# sess = tf.InteractiveSession()
# all_img = sess.run(test)
# print(np.array(all_img).shape)

# row = im.shape[0]
# column = im.shape[1]
# im = im.flatten().reshape(row, column)
#
# plt.imshow(im[::], cmap="gray")
# plt.show()
