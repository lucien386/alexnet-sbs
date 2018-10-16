from layer import *
import tensorflow as tf
import numpy as np
import os
import glob


# this depends on your own file path!
path = "/Users/frank/Documents/GitHub/alexnet-sbs/dataSet/parsed_train/*.npy"
fnames = glob.glob(path)
#test_path = "D:/alexnet-sbs/dataSet//*.npy"
class batch_helper():
    def __init__(self):
        self.batches = []
        #self.test = 

    def load_data(self):
        for filename in fnames:
            print('Loading ' + filename)
            self.batches.append(np.load(filename))

    def next_batch(self):
        return self.batches.pop()

# BUILD NET
pred = alex_net(x_input, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = pred))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# TEST NETWORK
matches = tf.equal(tf.argmax(pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feeder = batch_helper()  # working on here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    feeder.load_data()
    for i in range(training_iters):
        batch = feeder.next_batch()
<<<<<<< HEAD
        cur_input = []
        cur_label = []
        for item in batch:
            cur_label.append(item[0])
            cur_input.append(item[1])
        sess.run(train, feed_dict={x_input: cur_input, y_true: cur_label, keep_prob: dropout})
=======
        print(batch[0])
        sess.run(train, feed_dict={x_input: batch[1], y_true: batch[0], keep_prob: dropout})
>>>>>>> 9eea64caf0e0f3333a4a6a844755821dca543d3f
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i % 100 == 0:
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            #acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            #print(sess.run(acc, feed_dict={x: feeder.test_images, y_true: feeder.test_labels, hold_prob: 1.0}))
            print('\n')