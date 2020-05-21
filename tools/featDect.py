# -*- coding: utf-8 -*-

import cv2
from PIL import ImageEnhance,Image
import numpy as np
path=r"F:\SZU-Prj\jiaonang\PipePrj\2058.jpg"
vpath=r"F:\SZU-Prj\jiaonang\video-data\camera_4\151601.mp4"

cap = cv2.VideoCapture(vpath)
cc = cap.get(7)
print(cc)

sift = cv2.xfeatures2d.SIFT_create()
fast = cv2.FastFeatureDetector_create()
#img=cv2.imread(path)



while(cap.isOpened()):
    ret, frame = cap.read()
    frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    kp = sift.detect(frame1, None)
    img2 = cv2.drawKeypoints(frame1, kp, None, color=(255, 0, 0))
    cv2.imshow("1",img2)

    #print("已读完")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





#gray=cv2.drawKeypoints(img_arr[:,:,::-1],kp1, None, color=(255,0,0))#,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#print(len(kp1))
#import numpy as np
#import cv2 as cv
#img = cv.imread(path)
#gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#sift = cv.xfeatures2d.SIFT_create()
#kp = sift.detect(gray,None)
#img=cv.drawKeypoints(gray,kp,img)
#cv.imwrite('sift_keypoints.jpg',img)
#cv.imshow("1",img)

