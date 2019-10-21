# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_data', one_hot = True)

#print (mnist_data.train.images.shape, mnist_data.train.labels.shape)

batch_size = 100 # 每次批量学习的样本数量
n_batch = mnist_data.train.num_examples // batch_size #计算全部训练数据要分成多少组

x_plc_hlder = tf.placeholder(tf.float32, [None, 784]) # 原始数据是28*28的点阵,展开后就是784长度的数组
y_plc_hlder = tf.placeholder(tf.float32, [None, 10])

#第1层隐含层，200个神经元
L1_NodeCount = 200
W1 = tf.Variable(tf.truncated_normal(shape=[784, L1_NodeCount], stddev=0.1))
B1 = tf.Variable(tf.zeros([L1_NodeCount]) + 0.1)
S1 = tf.matmul(x_plc_hlder, W1)
C1 = tf.nn.tanh(S1 + B1)

#第2层隐含层，200个神经元
L2_NodeCount = 200
W2 = tf.Variable(tf.truncated_normal([L1_NodeCount, L2_NodeCount], stddev=0.1))
B2 = tf.Variable(tf.zeros([L2_NodeCount]) + 0.1)
S2 = tf.matmul(C1, W2)
C2 = tf.nn.tanh(S2 + B2)

#输出层，10个神经元，因为数字0-9共10个类别
L3_NodeCount = 10
W3 = tf.Variable(tf.truncated_normal([L2_NodeCount, L3_NodeCount], stddev=0.1))
B3 = tf.Variable(tf.zeros([L3_NodeCount]) + 0.1)
S3 = tf.matmul(C2, W3)
C3 = tf.nn.softmax(S3 + B3)

#
#loss = tf.reduce_mean(tf.square(y_plc_hlder - C1)) #2次代价函数
#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_plc_hlder, logits=C2)) #激活函数是sigmoid的交叉熵代价
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels= y_plc_hlder, logits= C3))#激活函数是softmax()的交叉熵代价
#
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

# tf.argmax(y_plc_hlder, 1)是返回一个列表中，值最大的位置，刚好用来解决one-hot型的数据
correct_prediction = tf.equal(tf.argmax(y_plc_hlder, 1), tf.argmax(C3, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# correct_prediction就是一个列表，记录每个学习样本得到的结果是否和预先的标签结果一致
# 例如一次学5个样本，就可能得到correct_prediction = [False  True False False False]
#accuracy0 = tf.cast([[False, True, False, False, False],
#                     [False, True, True, False, False]], tf.float32)
#accuracy0转化为[0, 1, 0, ...]的列表，True转化为1，False转化为0
#accuracy_rate = tf.reduce_mean(accuracy0)
#

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(301):
        for batch in range(n_batch):
            batch_x, batch_y = mnist_data.train.next_batch(batch_size)
            result = sess.run(train_step, feed_dict={x_plc_hlder:batch_x, y_plc_hlder:batch_y})

        if (epoch%10==0):
            #result = sess.run(C2, feed_dict={x_plc_hlder: batch_x})

            #result = sess.run(correct_prediction, feed_dict={x_plc_hlder:batch_x, y_plc_hlder:batch_y})
            #print("correct_prediction", result)
            acc_rate = sess.run(accuracy,\
                            feed_dict={x_plc_hlder:mnist_data.train.images,\
                            y_plc_hlder:mnist_data.train.labels})
            print("Iter:", str(epoch), ", Test accuracy rate:", acc_rate)

