from layer_hf import conv2d
from layer_hf import norm
from layer_hf import max_pool
import tensorflow as tf
import numpy as np

# Constants
learning_rate = 0.001
# training_iters = 200000
batch_size = 100 # 100 image at a time
display_step = 20 # Output frequency
image_size = 273
n_input = image_size**2 # 100*100
n_classes = 2 # Illed or not


# PlaceHolder
x_input = tf.placeholder(tf.types.float32, [None, n_input])
y_true = tf.placeholder(tf.types.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.types.float32)


def alex_net(input_x, weights, biases, dropout):
    input_x = tf.reshape(input_x, shape=[-1, image_size, image_size, 1])

    # First Layer
    w, b = weights["wc1"], biases["bc1"]
    conv1 = conv2d(input, w, b, strides=[1, 4, 4, 1])
    pool1 = max_pool(conv1)

    # Second Layer
    w, b = weights["wc2"], biases["bc2"]
    conv2 = conv2d(pool1, w, b)
    pool2 = max_pool(conv2)

    # Third Layer
    w, b = weights["wc3"], biases["bc3"]
    conv3 = conv2d(pool2, w, b)

    # Fourth Layer
    w, b = weights["wc4"], biases["bc4"]
    conv4 = conv2d(conv3, w, b)

    # Fifth Layer
    w, b = weights["wc5"], biases["bc5"]
    conv5 = conv2d(conv4, w, b)
    pool5 = max_pool(conv5)

    # Stretch pool5 into a long vector
    shape = [-1, int(np.prod(pool5.get_shape()[1:]))]
    vector = tf.reshape(pool5, shape)

    # Sixth Layer (Fully connected)
    w, b = weights["wn6"], biases["bn6"]
    norm6 = norm(vector, w, b)

    # Seventh Layer (Fully connected)
    w, b = weights["wn7"], biases["bn7"]
    norm7 = norm(norm6, w, b)

    # Eighth Layer (Ending Stage)
    w, b = weights["wn8"], biases["bn8"]
    end = norm(norm7, w, b, is_end=True)
    end = tf.nn.softmax(end)

    return end


weights = {
    "wc1": tf.Variable(tf.truncated_normal([11, 11, 1, 96],     stddev=0.01), name="wc1"),
    "wc2": tf.Variable(tf.truncated_normal([5, 5, 96, 256],     stddev=0.01), name="wc2"),
    "wc3": tf.Variable(tf.truncated_normal([3, 3, 256, 384],    stddev=0.01), name="wc3"),
    "wc4": tf.Variable(tf.truncated_normal([3, 3, 384, 384],    stddev=0.01), name="wc4"),
    "wc5": tf.Variable(tf.truncated_normal([3, 3, 384, 256],    stddev=0.01), name="wc5"),
    "wn6": tf.Variable(tf.truncated_normal([6*6*256, 4096],   stddev=0.01), name="wn1"),
    "wn7": tf.Variable(tf.truncated_normal([4096, 4096],        stddev=0.01), name="wn2"),
    "wn8": tf.Variable(tf.truncated_normal([4096, n_classes],   stddev=0.01), name="wn3")
}


biases = {
    "bc1": tf.Variable(tf.constant(0.0, shape=[96]),        name="bc1"),
    "bc2": tf.Variable(tf.constant(1.0, shape=[256]),       name="bc2"),
    "bc3": tf.Variable(tf.constant(0.0, shape=[384]),       name="bc3"),
    "bc4": tf.Variable(tf.constant(1.0, shape=[384]),       name="bc4"),
    "bc5": tf.Variable(tf.constant(1.0, shape=[256]),       name="bc5"),
    "bn6": tf.Variable(tf.constant(1.0, shape=[4096]),      name="bn1"),
    "bn7": tf.Variable(tf.constant(1.0, shape=[4096]),      name="bn2"),
    "bn8": tf.Variable(tf.constant(1.0, shape=[n_classes]), name="bn3")
}