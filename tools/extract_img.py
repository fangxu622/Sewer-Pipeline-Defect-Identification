import numpy as np
import cv2
import os
import shutil as sh
import string
import random


def random_char(y=5):
    return ''.join(random.choice(string.ascii_letters) for x in range(y))

def main(path="./video.mp4",feq=2,prestr="",out_dir="./out_dir"):
    if os.path.exists(out_dir):
        sh.rmtree(path=out_dir)
    os.mkdir(out_dir)
    #prestr=prestr+"_"+random_char()
    c=0
    cap = cv2.VideoCapture(path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        c = c + 1
        if not ret:
            break
        if c % feq==0 and c>200:
            #gray=frame[:,400:1500,:]
            gray=frame[:,430:1580,:]
            #gray_re=cv2.resize(gray,(0,0),interpolation=cv2.INTER_LINEAR)
            fn_tmp=os.path.join(out_dir,"ff"+str(int((c-100)/feq))+".jpg")
            cv2.imwrite(fn_tmp, gray)
        if c % 200==0:
            print(c)
    cap.release()


if __name__ == "__main__":
    import argparse
    #安装相关使用库
    
    #使用示例 
    #python --path F:\\SZU-Prj\\jiaonang\\video-data\\video.mp4 --out_dir ./out_dir
    #使用方式
    #--path 为视频路径 F:\\SZU-Prj\\jiaonang\\video-data\\video.mp4 or  F:/SZU-Prj/jiaonang/video-data/video.mp4
    #--feq 截取图片间隔
    #--out_dir 输出图像文件夹路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str,default=r"F:\SZU-Prj\jiaonang\2019_310_capsule_prj\detect-damage\data\1-Y1864-Y1879-20190304_102002.mp4")
    parser.add_argument("--feq",type=int,default=1)
    parser.add_argument("--out_dir",type=str,default=r"F:\SZU-Prj\jiaonang\2019_310_capsule_prj\detect-damage\data\data1-gt1")
    #parser.add_argument("--prestr",type=str,default="./out_dir")
    args = parser.parse_args()

    path=args.path
    feq=args.feq
    out_dir=args.out_dir

    main(path=path,feq=feq,out_dir=out_dir)
    
    print(args.path)
    
    