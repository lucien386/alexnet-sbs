import tensorflow as tf

# All the functions that help create layers should be defined here
# The actual creation of each layer should be done in layer.py


def conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(x, w, strides=strides, padding=padding)


# 2D convolution
def conv2d(input, w, b, strides = [1, 1, 1, 1]):
    conv = tf.nn.conv2d(input, w, strides=strides, padding="SAME", name="conv1")
    conv = tf.nn.bias_add(conv, b)
    conv = tf.nn.relu(conv)
    conv = tf.nn.local_response_normalization(conv, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
    conv = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    return conv


# Pooling
def max_pool(input):
    pool = tf.nn.local_response_normalization(input, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
    pool = tf.nn.max_pool(pool, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    return pool


# Normal
def norm(input, w, b, keep_prob = 0.5, is_end = False):
    fc1 = fc_layer(input, w, b)
    if not is_end:
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, keep_prob = keep_prob)
    return fc1


def fc_layer(input, w, b):
    fc = tf.nn.bias_add(tf.matmul(input, w), b)
    return fc


