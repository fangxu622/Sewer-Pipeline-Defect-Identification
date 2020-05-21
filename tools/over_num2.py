import numpy as np
import shutil as sh
import os

origin_path=r"E:\Python\PipePrj\ImageResult\489424063"
dest=r"E:\Python\PipePrj\ImageResult\over_num_hamm"

b=np.loadtxt(r'E:\Python\PipePrj\similarity_detect\hamm.txt',dtype=np.int32)

bb=b[1800:7600,1].mean()

print(bb)
c=0
scale_parse=5
# 移动图片去特征点大于指定数量的结果

thresh =11 #5*bb
c=0
for x in list(range(1800,7600)):
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
