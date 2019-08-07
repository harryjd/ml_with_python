# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from cv2 import waitKey
# from astropy.table.bst import MinValue
from sklearn.svm import SVC
import scipy
from skimage.feature import local_binary_pattern
import skimage.io as io

import hjdcv2common as hjdcv2

def ShowImg(title, imgdata):
    cv2.imshow(title, imgdata)
    cv2.waitKey()
    cv2.destroyWindow(title)


#[-0.45942363 - 0.28265783  0.90180227 - 0.68954346  1.00925245  0.09804619]
####################################################################
#
####################################################################
def Call_of_Gen_Whole_LBP(src_dir_arg, dest_dir_arg,
                    nStart, nStop, file_tag="train", R=1, P=8):
    sum = 0
    block_sizes = [8, 16, 32, 64]#定义不同的block size
    features_count = 0
    uniform_lbp_bins = 10
    for i in range(len(block_sizes)):
        feature_count_i = (128//block_sizes[i])**2*uniform_lbp_bins  # 分区尺寸0下的特征数量
        features_count += feature_count_i
    #feature_count1 = (128//block_sizes[1])**2*uniform_lbp_bins  # 分区尺寸1下的特征数量
    #feature_count2 = (128//block_sizes[2])**2*uniform_lbp_bins  # 分区尺寸2下的特征数量
    
    #myHist_Whole记录一个文件不同分区大小下，全部的特征
    myHist_Whole = np.ndarray([(nStop - nStart + 1), features_count])

    for img_num in range(nStart, nStop + 1):
        #依次打开每个文件
        filepath = src_dir_arg + "/" + file_tag + str(img_num) + ".jpg"
        dest_filepath_uniform = dest_dir_arg + "/" + file_tag + \
            str(img_num) + "_sci_uniform_" + str(R) + "x" + str(P) + ".jpg"
        try:
            img = cv2.imread(filepath, 0)
            #进行LBP特征提取
            uniform_lbp_rxp = local_binary_pattern(img.copy(), P, R, method="uniform")
            #按不同的block_size来统计直方图
            for block_size_seq in range(len(block_sizes)):
                #按每个block_size级别计算直方图，作为特征向量
                # 计算横向、纵向各有多少个block
                x_count = img.shape[1]//block_sizes[block_size_seq]
                y_count = img.shape[0]//block_sizes[block_size_seq]
                #为当前block_size的特征向量申请存储的空间
                myHist = np.zeros([x_count*y_count, uniform_lbp_bins])
                block_seq = 0
                for y_pos in range(y_count):
                    for x_pos in range(x_count):
                        y0 = y_pos*block_sizes[block_size_seq]
                        y1 = y0 + block_sizes[block_size_seq]
                        x0 = x_pos*block_sizes[block_size_seq]
                        x1 = x0 + block_sizes[block_size_seq]
                        fv_r = np.histogram(uniform_lbp_rxp[y0:y1, x0:x1],\
                                          bins=np.arange(0, P+3)) #bins的数量必须要比实际数量多1，避免最后两个bin的数据合计
                        #fv_r是一个2维矩阵
                        # 把当前block的直方图展开，作为特征值写入一行
                        myHist[block_seq, :] = fv_r[0]
                        block_seq += 1
                #这里，myHist[,]存储了整个图的当前分区大小条件下，分区统计直方图，每行为1个分区
                if(block_size_seq == 0):
                    myHist_block_size = np.hstack((myHist.ravel())) #把lbp的分block直方图展平后，横向堆叠
                else:
                    myHist_block_size = np.hstack((myHist_block_size, myHist.ravel()))#把lbp的分block直方图展平后，横向堆叠
            #把myHist_block_size记录的当前图片的不同blocksize条件下的特征值，写到myHist_Whole的一行
            myHist_Whole[img_num-1, :] = myHist_block_size.ravel()
            print("img" +str(img_num) + " is finnished.")
        except:
            print("error in " + filepath)
    return myHist_Whole

if __name__ == "__main__":
    train_amount = 10  #训练样本有12个
    test_amount = 6    #验证样本有5个

    Label_0 = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
    Label = Label_0.T
    Label_1 = np.array([1, -1, 1, -1, 1, -1])
    Label_Test = Label_1.T   # 测试集的标签
    #print("Label", Label)
    #print("Label.shape", Label.shape)
    src_dir = r"images/dogcat"
    dest_dir = r"images/dogcat_dest"
    
    #提取指定图片的特征
    X_train = Call_of_Gen_Whole_LBP(
        src_dir, dest_dir, 1, 10, "train", R=1, P=8)
    X_test = Call_of_Gen_Whole_LBP(src_dir, dest_dir, 1, 6, "test", R=1, P=8)

    print("X_train.shape", X_train.shape)
    print("Label.shape", Label.shape)
    clf = SVC(kernel="linear").fit(X_train, Label)  # 训练SVM模型:linear, rbf,
    P = clf.decision_function(X_test)
    print(P)

    print("finished")
