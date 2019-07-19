# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
#from cv2 import waitKey
import hjdcv2common as hjdcv2
# from astropy.table.bst import MinValue



def ShowImg(title, imgdata):
    cv2.imshow(title, imgdata)
    cv2.waitKey()
    cv2.destroyWindow(title)

'''
####################################################################
#
####################################################################
'''
#def call_of_lbp(src_file):

def Call_of_Read_IMG(src_dir_arg, dest_dir_arg):
    #list = os.listdir(src_dir_arg)
    sum = 0
    for i in range(1, 5):
        filepath = src_dir_arg + "/" + str(i) + ".png"
        dest_filepath = dest_dir_arg + "/" + str(i) + "_lbp_rotate.png"
        dest_filepath_basic = dest_dir_arg + "/" + str(i) + "_lbp.png"
        dest_filepath_circular = dest_dir_arg + "/" + str(i) + "_lbp_circularAreaBilinear1x8.png"
        dest_filepath_circular_bilinear = dest_dir_arg + "/test" + str(i) + "_lbp_circularBiLinear1x8.png"# 
        try:            
            img = cv2.imread(filepath, 0)
            #开始进行各种方法的LBP特征提取
            #res_rotate = hjdcv2.LBP_Rotate(img_copy) #LBP_Rotate(img.copy()) #LBP_Basic(img.copy())
            #res_basic = hjdcv2.LBP_Basic(img.copy())
            
            res_circular = hjdcv2.LBP_Circle_AreaBiLinear(img.copy(), 1, 8) #按分割的矩形面积进行插值的LBP，并进行适应取值
            #把LBP计算后的结果写成图片
            cv2.imwrite(dest_filepath_circular, res_circular)

            #res_circularBiLinear =  hjdcv2.LBP_Circle_BiLinear(img.copy(), 1, 8)
            #cv2.imwrite(dest_filepath_circular_bilinear, res_circularBiLinear)
            # resA = res.flatten()
            # print("Flattened :\n", resA.shape)
            #计算直方图
            
            '''
            myHist1 = cv2.calcHist([res_circular[0:99, 0:99]], [0], None, [256], [0,256])
            myHist2 = cv2.calcHist([res_circular[0:99, 100:199]], [0], None, [256], [0,256])
            myHist3 = cv2.calcHist([res_circular[0:99, 200:299]], [0], None, [256], [0,256])
            myHist4 = cv2.calcHist([res_circular[100:199, 0:99]], [0], None, [256], [0,256])
            myHist5 = cv2.calcHist([res_circular[100:199, 100:199]], [0], None, [256], [0,256])
            myHist6 = cv2.calcHist([res_circular[100:199, 200:299]], [0], None, [256], [0,256])
            myHist = np.vstack((myHist1, myHist2, myHist3, myHist4, myHist5, myHist6))
            print("myHist.shape", myHist.shape)
            '''

            '''
            plt.subplot(3, 1, 3*i-2),plt.title(str(i) + "Hist1.png")
            plt.xlabel("Bins"),plt.ylabel("# of Pixels"),plt.xlim([0, 256])
            plt.plot(myHist1)
            plt.subplot(3, 1, 3*i-1),plt.title(str(i) + "Hist2.png")
            plt.xlabel("Bins"),plt.ylabel("# of Pixels"),plt.xlim([0, 256])
            plt.plot(myHist2)
            plt.subplot(3, 1, 3*i),plt.title(str(i) + "Hist3.png")
            plt.xlabel("Bins"),plt.ylabel("# of Pixels"),plt.xlim([0, 256])
            plt.plot(myHist3)
            '''
                        
            sum = int(sum) + 1
            
            print(str(i) + " is finnished, number is " + str(sum))
        except:
            print( "error in " + filepath)
    #plt.show()
    

Label_0 = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
Label = Label_0.T
print("Label", Label)
print("Label.shape", Label.shape)
src_dir = r"images/dogcat"
dest_dir = r"images/dogcat_dest"
Call_of_Read_IMG(src_dir, dest_dir)

#hjdcv2.Tran_Sobel(src_dir, dest_dir)

'''
lbp_img1 = cv2.imread(dest_dir+"/1_lbp.png", 1)
lbp_img2 = cv2.imread(dest_dir+"/8_lbp.png", 1)
myHist1 = cv2.calcHist([lbp_img1], [0], None, [256], [0, 256])
myHist2 = cv2.calcHist([lbp_img2], [0], None, [256], [0, 256])
plt.subplot(2,1,1)
plt.title(str(1) + ".png")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0, 256])
plt.plot(myHist1)
plt.subplot(2,1,2),plt.xlim([0, 256]),plt.plot(myHist2)
plt.show()
print(myHist1.shape)
cv2.imshow("lbp_img", lbp_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#
'''
# plot
rX=[1,2,3]
rY=[3,4,8]
plt.subplot(2,2,1),plt.plot(rX, rY, "o")
plt.subplot(2,2,3),plt.plot(rX, rY, "o")
plt.show()
'''

# cv2.namedWindow("NamedWnd")

'''
filter_width, filter_height =3, 3

imgdir =  r"images/dogcat/"
for picNo in range(1, 2):
    imgfilepath = imgdir + str(picNo) + ".png"
    grayimg1 = cv2.imread(imgfilepath, 0)  # 1:读入RGB数据；0：读入灰度图
    srcimg_Height, srcimg_Width = grayimg1.shape[:2]
    # print('srcimg_Height, width', srcimg_Height, srcimg_Width)
    cv2.imshow('gray_img', grayimg1)
    # 
    mat_COV1 = np.zeros(shape=[srcimg_Height-filter_width+1, srcimg_Width-filter_width+1])
    mat1 = grayimg1[0:filter_height, 0:filter_width]
    linear_mat = np.reshape(mat1, (mat1.size, ))#把二维数据展开成1维的，方便求和
    print(linear_mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#计算LBP的值
P = 8
R=1
bx,by = 1,1
col= int(0.5 + bx + math.cos(2*math.pi*1/P)*R)
print('new col=', col)
col= int(0.5 + 2.6)
print('new col=', col)
'''
    
''' 按给定的模版图片的内容，在目标图片里找匹配上的内容，并画方框 '''
'''
imgfilename =  r"images/lena.jpg"  # r"images/book_cat_small.jpg" # r'./images/circle_gray.jpg'
tmpltfilename = r"images/lena_face.jpg"
img1 = cv2.imread(imgfilename, 1)  # 1:读入RGB；0:按灰度读入
imggray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)  # 转成灰度图更好识别

tmpltimg = cv2.imread(tmpltfilename, 0) # 读入模版图片，并且是按灰度读入
cv2.imshow('tmpltimg', tmpltimg)

cv2.imshow('imggray', imggray)

print('imggray.shape=', imggray.shape, 'tmpltimg.shape=', tmpltimg.shape)

myMethods = [cv2.TM_CCOEFF, cv2.TM_CCORR, cv2.TM_SQDIFF,
             cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]

res = cv2.matchTemplate(imggray, tmpltimg, method=myMethods[5])  #cv2.TM_SQDIFF_NORMED)
print('===res.shape:', res.shape)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#根据min_loc计算用于标注位置的方框
top_left = min_loc
bottom_right = (top_left[0] + tmpltimg.shape[1], top_left[1] + tmpltimg.shape[0])
cv2.rectangle(imggray, top_left, bottom_right, 255, 2)
plt.subplot(1,2,1), plt.imshow(res, cmap='gray')
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.subplot(1,2,2), plt.imshow(imggray, cmap='gray')
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.suptitle('cv2.TM_SQDIFF_NORMED')
plt.show()
'''