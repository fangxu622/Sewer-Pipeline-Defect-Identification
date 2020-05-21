
import cv2
import sys
sys.path.append(r"E:\Python\PipePrj\similarity_detect")
from dHash import DHash
import PIL.Image as Image
import numpy as np
import os
from matplotlib import pyplot as plt
path=r"E:\Python\PipePrj\ImageResult\489424063"

filename=os.listdir(path);
filelist=[x[0:x.index('.')] for x in filename]
filelist=list(map(int, filelist))
filelist.sort()
filelist.append(filelist[-1]+1)

data=np.zeros((filelist[-1],2))

scale_parse=5
for x in list(range(1,filelist[-1],scale_parse)):
    #if x % 100==0:
        #print(x)
    if x+scale_parse>=filelist[-1]:
        print("out index ,over!")
        break
    img_path1 = os.path.join(path, str(x) + ".jpg")
    img1=Image.open(img_path1)
    img_path2 = os.path.join(path, str(x+scale_parse) + ".jpg")
    img2 = Image.open(img_path2)

    hash1 = DHash.calculate_hash(img1)
    hash2 = DHash.calculate_hash(img2)
    hamming_distance = DHash.hamming_distance(hash1, hash2)

    data[x, 0] = x
    data[x,1]=hamming_distance

np.savetxt("hamm.txt", data,fmt="%d")

plt.scatter(data[:,0],data[:,1],s = 0.5)

plt.show()


