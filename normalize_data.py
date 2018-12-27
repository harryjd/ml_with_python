import pandas as pd
import numpy as np

# ========================================================================
# 进行数据的归一化
def minmax(dataSet):
    minDf = dataSet.min()
    maxDf = dataSet.max()
    normSet = (dataSet- minDf)/(maxDf-minDf)
    return normSet

bcw_data = pd.read_csv('csv/bcw_data_600.csv')
# bcw_data_arr = bcw_data.values             # 读取的数据转化成numpy的矩阵numpy.ndarray

## 利用原始数据的数值部分，归一化，然后和分类标签拼接成完整的数据
bcw_data_normal0 = pd.concat([bcw_data.iloc[:,0], minmax(bcw_data.iloc[:, 1:10]), bcw_data.iloc[:, -1]], axis=1)

# print('bcw_data:======\n', bcw_data.iloc[0,:])
# print('bcw_data_normal:======\n', bcw_data_normal0.iloc[0,:])
bcw_data_normal0.to_csv("csv/bcw_data_normal600.csv", index=0)   # index=0代表不保存行索引