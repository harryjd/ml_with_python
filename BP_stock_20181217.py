import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mplt


# ========================================================================
# 进行数据的归一化
def func_minmax(dataSet):
    minDf = dataSet.min()
    maxDf = dataSet.max()
    normSet = (dataSet - minDf) / (maxDf - minDf)
    return normSet


date = np.linspace(1, 5, 5)  # 从1到15，平均距离产生15个数

beginPrice = np.array([[15.90, 16.26, 10.1, 19.19, 25.33]])
endPrice = np.array([[25.90, 26.26, 15.90, 23.19, 20.33]])

'''
mplt.figure()
# draw line one by one in different color
for i in range(0, 5):
    dateOne = np.zeros([2])
    dateOne[0] = i;
    dateOne[1] = i;
    priceOne = np.zeros([2])
    priceOne[0] = beginPrice[i]
    priceOne[1] = endPrice[i]
    if endPrice[i] > beginPrice[i]:
        mplt.plot(dateOne, priceOne, 'r', lw=6)      # draw line one by one in different color
    else:
        mplt.plot(dateOne, priceOne, 'g', lw=6)
mplt.show()
'''

normalPrice = func_minmax(endPrice)  # 1行5列
dateNormal = np.zeros([5, 1])
priceNormal = np.zeros([5, 1])  # 归一化后的数据

# 归一化日期，以便用于输入
for i in range(0, 5):
    dateNormal[i, 0] = i / 4.0

priceNormal = normalPrice.T  # 如果normalPrice是一维的，就需要增加维度，方法为：[:, np.newaxis]

# x_data = np.linspace(-0.5, 0.5, 50)[:, np.newaxis] #5 rows, 1 column,增加一个新的维度

X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_uniform([1, 5], 0, 1))
B1 = tf.Variable(tf.random_uniform([1, 5], 0, 1))
WB1 = tf.matmul(X, W1) + B1
NN_Layer1 = tf.nn.relu(WB1)      # 激励函数是relu

W2 = tf.Variable(tf.random_uniform([5, 1], 0, 1))
B2 = tf.Variable(tf.random_uniform([5, 1], 0, 1))
WB2 = tf.matmul(NN_Layer1, W2) + B2
NN_Layer2 = tf.nn.relu(WB2)      # 激励函数是relu

loss = tf.reduce_mean(tf.square(Y-NN_Layer2))      #
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for round in range(0, 200):
    # print('WB1:\n', sess.run(WB1, feed_dict={X:dateNormal}))
    # print('NN_Layer2:\n', sess.run(NN_Layer2, feed_dict={X: dateNormal}))
    # print('loss:\n', sess.run(loss, feed_dict={X:dateNormal, Y:priceNormal}))
    sess.run(train, feed_dict={X:dateNormal, Y:priceNormal})
print(sess.run(loss, feed_dict={X:dateNormal, Y:priceNormal}))
sess.close()
