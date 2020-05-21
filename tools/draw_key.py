# -*- coding: utf-8 -*-

import cv2

path=r"E:\Python\PipePrj\over80\2527.jpg"
sift = cv2.xfeatures2d.SIFT_create()

img=cv2.imread(path)

kp1,des1= sift.detectAndCompute(img,None)

gray=cv2.drawKeypoints(img,kp1,img)#,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("1",gray)

print(len(kp1))
#import numpy as np
#import cv2 as cv
#img = cv.imread(path)
#gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#sift = cv.xfeatures2d.SIFT_create()
#kp = sift.detect(gray,None)
#img=cv.drawKeypoints(gray,kp,img)
#cv.imwrite('sift_keypoints.jpg',img)
#cv.imshow("1",img)

cv2.waitKey(0)