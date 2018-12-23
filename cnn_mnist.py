import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_data', one_hot = True)

print (mnist_data.train.images.shape, mnist_data.train.labels.shape)

imageInput = tf.placeholder(tf.float32, [None, 784])   # 原始数据是28*28的点阵
labelInput = tf.placeholder(tf.float32, [None, 10])

# 把输入数据调整一下形状:[-1, 28, 28, 1]
# 第1个数字：经过形状调整后的数据，-1表示没有剩余？？？
# 第2个数字：数据的宽度；第3个数字，数据的高度
# 第4个数字：1---1个通道，代表灰度图
imageInputReshape = tf.reshape(imageInput, [-1, 28, 28, 1])

# 开始卷积处理
# w0表示卷积内核
# 高5，宽5，1个通道，结果用32维度的数字
W0 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev = 0.1))
b0 = tf.Variable(tf.constant(0.1, shape=[32]))

# 卷积的第1层，神经元的总输入不再是matmul的运算，而是conv2d卷积运算
layer1 = tf.nn.relu(
    tf.nn.conv2d(imageInputReshape, W0, strides=[1,1,1,1], padding='SAME')+b0)
# 添加池化层，进行抽样，简化运算
# ksize=[1,4,4,1]，进行下采样，数据量大大减少，原始数据维度对应除以ksize的元素，成为[-1,7,7,1]
layer1_pool = tf.nn.max_pool(layer1, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

# 定义输出层,有1024个输出节点
W1 = tf.Variable(tf.truncated_normal([7*7*32, 1024], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_reshape = tf.reshape(layer1_pool, [-1, 7*7*32]) # 和W1的形状有关，把4维数据再转换为2维数据
h1 = tf.nn.relu(tf.matmul(h_reshape, W1)+b1)

#
W2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))

predict = tf.nn.softmax(tf.matmul(h1, W2) + b2)
# predict 就是 N行10列

# 再来考虑误差的计算方法，然后训练，不断减少误差
# 标签数据也是 N行10列，每一行10个数字，OneHot形式，来表示是哪个数字
loss0 = labelInput*tf.log(predict)
loss1 = 0
for m in range(0, 100):  # 每次训练10张图
    for n in range(0, 10):
        loss1 = loss1 - loss0[m, n]
loss = loss1 /100

# 训练的定义
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# 开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        images, labels = mnist_data.train.next_batch(500)
        sess.run(train, feed_dict={imageInput:images, labelInput:labels})
        pred_test = sess.run(predict, feed_dict={imageInput:images, labelInput:labels})
        acc = tf.equal(tf.argmax(pred_test, 1),
                       tf.argmax(mnist_data.test.labels, 1))
        acc_float = tf.reduce_mean(tf.cast(acc, tf.float32))
        acc_result = sess.run(acc_float,
                              feed_dict={imageInput:mnist_data.test.images,
                                         labelInput:mnist_data.test.labels})
        print(acc_result)