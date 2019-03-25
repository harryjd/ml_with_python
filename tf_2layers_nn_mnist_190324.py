# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_data', one_hot = True)

# print (mnist_data.train.images.shape, mnist_data.train.labels.shape)

batch_size = 50 # 每次批量学习的样本数量
n_batch = mnist_data.train.num_examples // batch_size

# imageInput = tf.placeholder(tf.float32, [None, 784])
# labelInput = tf.placeholder(tf.float32, [None, 10])
x_plc_hlder = tf.placeholder(tf.float32, [None, 784]) # 原始数据是28*28的点阵,展开后就是784长度的数组
y_plc_hlder = tf.placeholder(tf.float32, [None, 10])

#
W1 = tf.Variable(tf.random_normal([784, 10]))
B1 = tf.Variable(tf.random_normal([10]))
S1 = tf.matmul(x_plc_hlder, W1)
C1 = tf.nn.softmax(S1 + B1)

W2 = tf.Variable(tf.random_normal([10, 10]))
B2 = tf.Variable(tf.random_normal([10]))
S2 = tf.matmul(C1, W2)
C2 = tf.nn.sigmoid(S2 + B2)
#
loss = tf.reduce_mean(tf.square(y_plc_hlder - C1))

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

#
correct_prediction = tf.equal(tf.argmax(y_plc_hlder, 1), tf.argmax(C1, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(301):
        for batch in range(n_batch):
            batch_x, batch_y = mnist_data.train.next_batch(batch_size)

            # result = sess.run(C1, feed_dict={x_plc_hlder:batch_x})
            # print("C1", result)
            # result = sess.run(C2, feed_dict={x_plc_hlder:batch_x})
            # print("C2", result)
            result = sess.run(train_step, feed_dict={x_plc_hlder:batch_x, y_plc_hlder:batch_y})

        if (epoch%10==0):
            #result = sess.run(C2, feed_dict={x_plc_hlder: batch_x})
            #print("C2", result)
            acc_rate = sess.run(accuracy, feed_dict={x_plc_hlder:batch_x, y_plc_hlder:batch_y})
            print("Iter:", str(epoch), ", Test accuracy rate:", acc_rate)
