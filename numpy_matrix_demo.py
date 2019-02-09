import numpy as np

# 关于numpy矩阵的使用方法
Z0 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Z1 = Z0[:,0]        # 如果只取1列，就会降低维度，成为(3,)！！！
Z2 = Z0[:, 0:2]     # 如果取多列，不会降低维度！
print('Z0\n', Z0)
print('Z1\n', Z1)
print('Z2\n', Z2)

print('Z0.shape=', Z0.shape, type(Z0))
print('Z1.shape=', Z1.shape, type(Z1))
print('Z2.shape=', Z2.shape, type(Z2))
