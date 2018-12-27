'''

'''
import numpy as np
import pandas as pd
import tensorflow as tf

print(' 准备从数据文件读入归一化数据 ')
bcw_data_file = r'csv/bcw_data_normal10.csv'   # r'bcw-train-10.csv'
bcw_data = pd.read_csv(bcw_data_file)          # 从文件读取数据
# bcw_data_arr1 = bcw_data.iloc[:,1:-1].values      # 读取的数据转化成numpy的矩阵numpy.ndarray
print('===')
bcw_data_arr2 = bcw_data.values      #

y = bcw_data_arr2[1, -1][np.newaxis, np.newaxis]
print('y=', y)

batch_size = 5
b2 = np.random.randn(1)
print('b2=', b2)
X_input = tf.placeholder(tf.float32, shape=[batch_size, 9])
Y_Label = tf.placeholder(tf.float32, shape=[batch_size, 1])

W1 = tf.Variable(tf.random_uniform([9, 3], 0, 1))      # 隐含层3个神经元
Belta1 = tf.Variable(tf.zeros([1, 3]) + 0.1)
W2 = tf.Variable(tf.random_uniform([3, 1], 0, 1))        #
Belta2 = tf.Variable(tf.zeros([1, 1])+0.1)

S1 = tf.matmul(X_input, W1) + Belta1    # 隐含层的总输入
B1 = tf.nn.relu(S1)

S2 = tf.matmul(B1, W2) + Belta2
B2 = tf.nn.relu(S2)    # 输出

loss = tf.reduce_mean(tf.square(Y_Label - B2))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

DataLen = bcw_data.shape[0]   # 计算数据的数量
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    row1 = 0
    for train_round in range(0, 10000):
        row1 = train_round*batch_size % DataLen
        row2 = row1 + batch_size

        X = bcw_data_arr2[row1:row2, 1:-1]   # [None,:]  # 没有最后一列标签
        # print('X=\n', X)
        # print('======')
        Y = (bcw_data_arr2[row1:row2, -1][None,:] + 0.0).T # [np.newaxis, np.newaxis]+0.0
        # print('Y_Label=\n', Y)
        # print('train_round=', train_round)
        # B1_out = sess.run(B1, feed_dict={X_input: X})
        # print('B1_out\n', B1_out)
        # B2_out = sess.run(B2, feed_dict={X_input: X})
        # print('B2_out\n', B2_out)
        loss_out = sess.run(loss, feed_dict={X_input: X, Y_Label: Y})
        sess.run(train_step, feed_dict={X_input: X, Y_Label: Y})

        if (train_round%200==0):
            print('loss =', loss_out)
            # print('W1\n', sess.run(W1))
            # print('W2\n', sess.run(W2))
    print('W1\n', sess.run(W1))
    print('W2\n', sess.run(W2))