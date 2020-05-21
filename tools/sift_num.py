
#生成sift特征点序列图，并对序列图像进行抽稀，均值聚类获取异常帧数，并取中值

import cv2
import numpy as np
import os
import sys
from matplotlib import pyplot as plt

# 定义旋转rotate函数
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated

path=r"E:\fxworkspace\PipePrj\xili_data\0129_1_6063"

filename=os.listdir(path);
filelist=[x[0:x.index('.')] for x in filename]
filelist=list(map(int, filelist))
filelist.sort()


sift = cv2.xfeatures2d.SIFT_create()
#f=open("sift_num.txt","w");
c=0
scale_cut = 5 # 裁剪因子
scale_parse=5 # 序列图像系数因子


data=np.zeros((filelist[-1]+1,2))
for x in filelist:
    c=c+1
    img_path=os.path.join(path,str(x)+".jpg")
    img = cv2.imread(img_path,0)
    img = img[img.shape[0] // scale_cut:img.shape[0] - img.shape[0] // scale_cut // 2,
          img.shape[1] // scale_cut:img.shape[1] - img.shape[1] // scale_cut // 2]
    #print(img.shape)
    #if img.shape()==
    #gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp1= sift.detect(img,None)
    #print(len(kp1))
    if c % 200 ==0:
        print(c)
    data[x, 0] =x
    data[x, 1]=len(kp1)

    #f.write(x[0:x.index(".")]+" "+str(len(kp1))+"\n")
np.savetxt("siftpoint.txt", data,fmt="%d")

plt.scatter(data[:,0],data[:,1],s = 0.5)

plt.show()








