# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 21:36:03 2018

@author: szu_dell
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


#df=pd.read_csv("result.txt",sep=" ")
#plt.scatter(df['0'],df['3'])

b=np.loadtxt(r'E:\Python\PipePrj\similarity_detect\hamm.txt',dtype=np.int32)
#plt.scatter(b[1800:7800,0],b[1800:7800,1],s = 0.5)
plt.scatter(b[2000:7500,0],b[2000:7500,1],s = 0.5)
#plt.xlim(1800,7800)
plt.xlim(2000,7500)
plt.ylim(0,150)
plt.show()


