from layer import *
import tensorflow as tf
import numpy as np
import os

training_iters = 200
directory = os.fsencode(path)
path = "/Users/frank/Documents/Github/alexnet-sbs/dataSet/train/normal_batch"

class batch_helper():
    def __init__(self):
        self.batches = []
        self.path = ''

    def load_data(self):
        for filename in os.listdir(path):
            self.batches.append(np.load(filename))

    def next_batch(self):
        return self.batches.pop()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feeder = batch_helper()  # working on here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for i in range(training_iters):
        batch = feeder.next_batch()
        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i % 10 == 0:
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            print(sess.run(acc, feed_dict={x: feeder.test_images, y_true: feeder.test_labels, hold_prob: 1.0}))
            print('\n')