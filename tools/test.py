
#保存图像

import numpy as np
import cv2

cap = cv2.VideoCapture(r"I:\20180129\1\20180129082515\1517214315489424063.h264")
c = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    c = c + 1
    if not ret:
        break
    if c % 1==0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray=gray[:,360:1590].copy()
        gray_re=res=cv2.resize(gray,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)

        cv2.imshow('frame', gray_re)
        cv2.imwrite(r'E:\Python\PipePrj\ImageResult\489424063\\' + str(c) + '.jpg', gray_re)  # 存储为图像

        if c % 200 == 0:
            print(c)

        #print("已读完")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()