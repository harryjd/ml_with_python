# -*- coding: utf-8 -*-
"""
# 实现对一个图片的数据，按指定的过滤窗口，滑动进行滤波处理。也是卷积操作的基础操作。
Created on Sun Feb 10 15:49:59 2019

@author: HarryJD
"""

import cv2
import numpy as np

def cv_show(img_data, wnd_name):
    cv2.imshow(wnd_name, img_data)
    cv2.waitKey(0)
    cv2.destroyWindow(wnd_name)
    
img_filename = r'images\cat.png'
img_color = cv2.imread(img_filename, 1)   #1:读入BGR三个通道的图像数据；0：读入灰度图
#img_gray  = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
#ret, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)

cv_show(img_color, 'Color Img')

mat_filter1 = np.array([[-1, -1, -1],
                        [ 0,  1,  0],
                        [ 1,  1,  1]])
filter1_width = 3
print('img size', img_color.shape)
width = img_color.shape[1]     #当前图片宽度300
height = img_color.shape[0]    #当前图片高度250
channel = img_color.shape[2]   #颜色通道数

row = 0      # 纵向起始的位置
col = 0      # 横向起始的位置
chnl = 0     # 颜色通道

mat_COV1 = np.zeros(shape=[height-filter1_width+1, width-filter1_width+1])
print(mat_COV1.shape)

for row in range(0, height - filter1_width + 1):
    for col in range(0, width - filter1_width + 1):
        sum_r = 0    # 三个通道的累积，初始化为0
        for chnl in range(0, channel):
            # 1 切片数据
            mat1 = img_color[row:row+filter1_width, col:col+filter1_width, chnl]
            mat_r1 = mat_filter1 * mat1
            sum_r1 = np.sum(np.reshape(mat_r1, (mat_r1.size,)))
            sum_r = sum_r + sum_r1
        #end of for chnl in range(0, channel)
        mat_COV1[row, col] = sum_r//3
#print(mat_COV1)
cv_show(mat_COV1, 'mat_COV1')
