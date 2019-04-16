import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_data', one_hot = True)
batch_size = 100 # 每次批量学习的样本数量
n_batch = mnist_data.train.num_examples // batch_size #计算全部训练数据要分成多少组

#初始化卷积核
def weight_variable(myShape):
    initial = tf.truncated_normal(myShape, stddev=0.1) #生成一个截断的正态分布
    return tf.Variable(initial)

def bias_variable(myShape):
    initial = tf.constant(0.1, shape=myShape)
    return tf.Variable(initial  )

#卷积层
def myConv2d(x, W):
    # strides的第0个、第2个参数都必须是1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#池化层
def max_pool_2x2(x):
    #kSize [1, x, y ,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x_plc_hlder = tf.placeholder(tf.float32, [None, 784]) # 原始数据是28*28的点阵,展开后就是784长度的数组
y_plc_hlder = tf.placeholder(tf.float32, [None, 10])

#把原来展开成1维的数据重新组织成4维的：[batch, int_height, int_width, channel] 28 * 28
Channels = 1 #灰度图，通道数为1
x_image = tf.reshape(x_plc_hlder, [-1, 28, 28, Channels])

W_conv1 = weight_variable([5, 5, Channels, 32]) #5 * 5的采样窗口，32个卷积核从1个平面(灰度图)提取特征
b_conv1 = bias_variable([32]) #每个卷积核一个偏置? 不是共享神经网络的参数吗？
#x_image与权值向量进行卷积，再加上偏置
h_conv1 = tf.nn.relu(myConv2d(x_image, W_conv1) + b_conv1) #卷积核有32个，因此得到32个卷积特征平面
h_pool1 = max_pool_2x2(h_conv1) #池化

W_conv2 = weight_variable([5, 5, 32, 64]) #5 * 5的采样窗口，64个卷积核从32个平面(灰度图)提取特征
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(myConv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#初始化第1个全连接层的权值
W_fc1 = weight_variable([7*7*64, 1024]) #上一个层池化后输出是7*7的二位特征，有64张7*7的平面
b_fc1 = bias_variable([1024])  #全连接层有1024个神经元，可以调整

#把池化层2的输出扁化成1维，才能输入分类器
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#keep_prob用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #按一定概率禁止神经元输出，避免过拟合

#初始化第2个全连接层的权值
W_fc2 = weight_variable([1024, 10]) #
b_fc2 = bias_variable([10])  #第2个全连接层有10个神经元，对应输出10个可能的标签

#计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#交叉熵代价函数
cross_entropy = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_plc_hlder, logits=prediction))
#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_plc_hlder, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    init_step = tf.global_variables_initializer()
    sess.run(init_step)
    for epoch in range(11):
        for batch in range(n_batch):
            batch_x, batch_y = mnist_data.train.next_batch(batch_size)
            result = sess.run(train_step, feed_dict={x_plc_hlder:batch_x, y_plc_hlder:batch_y, keep_prob:0.75})

        acc_rate = sess.run(accuracy,\
                            feed_dict={x_plc_hlder:mnist_data.test.images,\
                            y_plc_hlder:mnist_data.test.labels})
        print("Iter:", str(epoch), ", Test accuracy rate:", acc_rate)
