'''
用Python实现BP神经网络
'''
import numpy as np
import pandas as pd
# import tensorflow as tf

# --------------------------------------------
# 激活函数
# 当derive为True时，是对sigmoid函数求一阶导数后的函数，f'=y*(1-y)
def f_sigmoid(x, derive=False):
    if not derive:
        return 1 / (1 + np.exp(-x))
    else:
        return x * (1 - x)  # 这里的x实际上是f(x)

print(' 准备从数据文件读入归一化数据 ')
bcw_data_file = r'bcw_data_normal100.csv'   # r'bcw_data_normal10.csv' r'bcw-train-10.csv'
bcw_data0 = pd.read_csv(bcw_data_file)           # 从文件读取数据
dataLen= bcw_data0.shape[0]                      # 样本数量
properties = bcw_data0.shape[1]
print('properties', properties)

bcw_data_arr0 = bcw_data0.values                 # 原始数据, P1,P2,...,P9,Label
bcw_data_arr1 = bcw_data_arr0[:,1:-1]  # np.concatenate((bcw_data_arr0[:,1:-1], np.ones([dataLen, 1])), axis=1)
# print('bcw_data_arr1\n', bcw_data_arr1[0:2,:])
# bcw_data_arr1 = bcw_data.iloc[:,1:-1].values      # 读取的数据转化成numpy的矩阵numpy.ndarray
print('===')
batch_size = 10      # 每次读N个数据
L1NodesNum = 10
L2NodesNum = 1
learnRate = 0.1
Y = bcw_data_arr0[:, -1][np.newaxis, np.newaxis].T

WA = np.random.randn(9, L1NodesNum)            # 不包含偏置的连接
WB = np.random.randn(L1NodesNum, L2NodesNum)   # 不包含偏置的连接
BetaA = np.random.randn(1, L1NodesNum)
BetaB = np.random.randn(1, L2NodesNum)
nTrainRound = 0
for nTrainRound in range(0, 100000):
    ## 1:前向计算
    row1 = (nTrainRound * batch_size)%dataLen
    row2 = row1 + batch_size
    inputX = bcw_data_arr1[row1:row2, :]
    # print('=== inputX: ===\n', inputX)
    SA = np.matmul(inputX, WA) + BetaA   # 第1层的总输入（含偏置）
    # print('===SA:\n', SA)
    CA = f_sigmoid(SA)                    # 第1层的输出
    # print('===CA:\n', CA)
    SB = np.matmul(CA, WB) + BetaB       # 第2层的总输入（含偏置）
    CB = f_sigmoid(SB)                    # 第2层的输出
    # print('=== CB\n', CB)
    YTrain = Y[row1:row2, -1]
    # print('=== YTrain: ===\n', YTrain)

    ## 根据误差计算W2的调整
    Diff_CB = YTrain - CB
    if(nTrainRound%100==0):
        MeanDiff_CB = np.mean(np.abs(Diff_CB))
        print('=== MeanDiff_CB:\n', MeanDiff_CB)

    Delta_CB = Diff_CB * f_sigmoid(CB, True)    # 对每个输出节点都一次完成一阶导函数计算.f'(x)=f(x)(1-f(x))
    # print('=== Delta_CB:\n', Delta_CB)
    # print('=== CA:\n', CA)
    # print('===原 WB:\n', WB)
    Delta_WB = np.matmul(CA.T, Delta_CB)  # 累积多个样本的误差
    Delta_BetaB = Delta_CB.sum(axis=0)   # 把各个行的值累加，即纵向累加
    # print('Delta_WB=\n', Delta_WB)
    WB = WB + learnRate * Delta_WB
    BetaB = BetaB + learnRate * Delta_BetaB
    # print('新W2=\n', W2)

    ## 根据误差计算W1的调整
    Delta_CA = np.matmul(Delta_CB, WB.T) * f_sigmoid(CA, True)
    # print('=== Delta_CA:\n', Delta_CA)
    # print('原来的W1:\n', W1)
    Delta_WA = np.matmul(inputX.T, Delta_CA)
    # print('=== Delta_WA:\n', Delta_WA)
    # print('=== WA:\n', WA)
    WA = WA + learnRate * Delta_WA
    Delta_BetaA = Delta_CA.sum(axis=0)
    BetaA = BetaA + learnRate * Delta_BetaA
    # print('调整后的W1:\n', W1)
    # Delta_WA = np.matmul(Delta_C1.T, inputX)*f_sigmoid(CA, derive= True)
