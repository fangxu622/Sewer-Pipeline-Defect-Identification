
## Overview
This is an implementation of  the paper :**Sewer Pipeline Fault Identification Using Anomaly Detection Algorithms on Video Sequences**

## Required packages:
+ python 3+
+ sklearn 0.20.3


## Data
+ data/dataset-1
+ data/video-1
+ data/video-2 
+ Dataset url 
   + BaiduYunPan：
  链接：https://pan.baidu.com/s/1Pi9K__knSjFcZoHz3I37SA 
    提取码：n2lc
   + Google Drive: 
   https://drive.google.com/drive/folders/17z7fOopeP1bf4jVx6eX-uoXg6Db8QC9N?usp=sharing


Feature csv file of all dataset have produced in "data" folder 

## usage

+ feature-extraction.py

    It will produce the csv file include all fetures

+ Anomaly-detection.py

    You can select different anomaly detection and feature groups.
    The result of all feature group and anomaly detection alogrithm will be default produced.
## Result
Detail in experiment-1 fold.

## Note

+ *Due to the randomness and probability value results of the initial value of the anomaly detection algorithm. The results of each run will have slight difference on probability values. But this does not affect the overall evaluation of the results of the algorithm.*


+ *If you don't want to copy and move image file accordding to the classificaiton,please comment the mv_result() function.comment for default.Otherwise it will take up a lot of storage space*

