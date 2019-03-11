# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt

# 取得给定的LBP字符串的最小二进制值，实现旋转不变形
def Rotate_LBP_Min(str_lbp_input):
    str_lbp_tmp = str_lbp_input
    MinValue = int(str_lbp_tmp, 2) #转换二进制数
    nLen = len(str_lbp_tmp)
    #暴力尝试方式取得最小值
    for npos in range(nLen):
        str_head = str_lbp_tmp[0]
        str_tail = str_lbp_tmp[1:]
        str_lbp_tmp = str_tail + str_head
        CurrentValue = int(str_lbp_tmp, 2)
        if CurrentValue<MinValue:
            MinValue = CurrentValue
    return MinValue

'''
# LBP_Rotate(image)实现LBP旋转不变性的获取特征
'''
def LBP_Rotate(image):
    H, W = image.shape[:2]      #获得图像长宽
    xx = [-1,  0,  1, 1, 1, 0, -1, -1]
    yy = [-1, -1, -1, 0, 1, 1,  1,  0]    #xx, yy 主要作用对应顺时针旋转时,相对中点的相对值.
    
    #创建0数组,显而易见维度原始图像的长宽分别减去2，并且类型一定的是uint8,无符号8位,opencv图片的存储格式.
    res = np.zeros(shape=(H-2, W-2), dtype="uint8")
    
    for row in range(1, H - 1):
        for col in range(1, W - 1):
            str_lbp_tmp = "" #拼接二进制字符串
            for m in range(0, 8):
                Xtemp = xx[m] + col   
                Ytemp = yy[m] + row    #分别获得对应坐标点
                #print("Ytemp, Xtemp", Ytemp, Xtemp)
                if image[Ytemp, Xtemp] > image[row, col]: #像素比较
                    str_lbp_tmp = str_lbp_tmp + '1'
                else:
                    str_lbp_tmp = str_lbp_tmp + '0'
            #print int(str_lbp_tmp, 2)
            
            res[row - 1][col - 1] =Rotate_LBP_Min(str_lbp_tmp)   #写入结果中
            #print("res[i][j]: i, j=", row - 1, col - 1)
            #print("res[row - 1][col - 1]", res[row - 1][col - 1])
    # print("res.shape=", res.shape[:2])
    return res


'''
# 主过程
'''
def Tran_LBP(src_dir_arg, dest_dir_arg):
    #list = os.listdir(src_dir_arg)
    sum = 0
    for i in range(1,11):
        filepath = src_dir_arg + "/" + str(i) + ".png"
        dest_filepath = dest_dir_arg + "/" + str(i) + "_lbp.png"
        try:            
            img = cv2.imread(filepath, 0)
            #cv2.imshow("temp", img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            #开始进行LBP特征提取
            res = LBP_Basic(img.copy()) #LBP_Rotate(img.copy()) #LBP_Basic(img.copy())
            resA = res.flatten()
            print("Flattened :\n", resA.shape)
            cv2.imwrite(dest_filepath, res)
            sum = int(sum) + 1
            print(str(i) + " is finnished, number is " + str(sum))
        except:
            print( "error in " + filepath)


src_dir = r"images/dogcat"
dest_dir = r"images/dogcat_dest"


Tran_LBP(src_dir, dest_dir)
