# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:55:06 2019

@author: harryjd
"""
import cv2
import matplotlib.pyplot as plt

img_filename = r'images\lena.png'
img_color = cv2.imread(img_filename, 1)
# cv2.imshow('color img', img_color)
colors = ('b', 'g', 'r')

for i, mycolor in enumerate(colors):
    # print('i=', i, 'color=', color)
    histr = cv2.calcHist([img_color], [i], None, [256], [0, 256])
    plt.plot(histr, color=mycolor)
    plt.xlim([-1, 256])  #xlim是设定横坐标范围

cv2.waitKey(0)
cv2.destroyAllWindows()
