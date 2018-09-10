import tensorflow as tf

# All the functions that help create layers should be defined here
# The actual creation of each layer should be done in layer.py

def conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(x, w, strides=strides, padding=padding)


def max_pool_2by2(x, ksize=[1, 2, 2, 2], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(x, ksize=ksize,
                          strides=strides, padding=padding)


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b
