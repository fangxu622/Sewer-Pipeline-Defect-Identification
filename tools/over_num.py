import numpy as np
import shutil as sh
import os

origin_path=r"E:\fxworkspace\PipePrj\xili_data\0129_1_6063"
dest=r"E:\fxworkspace\PipePrj\ImageResult\overSiftNum"

b=np.loadtxt(r'E:\fxworkspace\PipePrj\siftpoint.txt',dtype=np.int32)

bb=b[1800:7800,1].mean()

print(bb)
c=0
scale_parse=5
# 移动图片去特征点大于指定数量的结果

thresh =bb
c=0
for x in list(range(2000,7800)):
    c = c + 1;
    if b[x, 1] > thresh:
        ori=os.path.join(origin_path,str(x)+".jpg")
        sh.copy(ori,dest)
    if c % 200==0:
        print(c)

# for x in range(2000,8000):
#     c=c+1;
#     if b[x,1] > 70:
#         ori=os.path.join(origin_path,str(x)+".jpg")
#         sh.copy(ori,dest)
#     if c % 200==0:
#         print(c)
