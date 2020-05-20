import numpy as np
import cv2
import argparse

Small_H_W=270
MATRIX_SIZE=15
img_fill_row = np.zeros((Small_H_W,Small_H_W,3),dtype=np.uint8)
prefix_str="capsule22_"

def img_fill():
    tem1=[]
    for jj in range(MATRIX_SIZE):
        tem1.append(img_fill_row.copy())
    #tem2=cv2.hconcat(tem1)
    return tem1

img_fill_col = img_fill()

#cap = cv2.VideoCapture(r"I:\20180129\1\20180129082515\1517214315489424063.h264")
#cap = cv2.VideoCapture(r"I:\20180129\0\20180129080032\1517212832974818707.h264");
#c = cap.get(7)#print(cc)

def matrix_image(m_path,dst_folder,EXTRACT_FREQUENCY):
    video = cv2.VideoCapture(m_path)
    if not video.isOpened():
        print("can not open the video")
        exit(1)
    tem_c=0
    matrix_count=0
    img_count=0
    image_list_row=[]
    image_list_col=[]
    while True:
        _,frame=video.read()
        tem_c=tem_c+1
        if tem_c % EXTRACT_FREQUENCY ==0:
            gray=frame[150:,360:1510,:]
            #tem_img0=color_img(gray)
            tem_img1=cv2.resize(gray,(Small_H_W,Small_H_W),
                            interpolation = cv2.INTER_LINEAR)
            image_list_row.append(tem_img1)
            if len(image_list_row) == MATRIX_SIZE:
                image_list_col.append(image_list_row)
                image_list_row=[]
            if len(image_list_col)== MATRIX_SIZE:
                tem_img4=merge_image(image_list_col)
                image_list_col=[]
                img_count=img_count+1
                filename=dst_folder+"/"+prefix_str+str(img_count)+".jpg"
                cv2.imwrite(filename,tem_img4)
        if frame is None:
            if not (len(image_list_col)==0 and len(image_list_row) ==0):
                for ii in range(MATRIX_SIZE-len(image_list_row)):
                    image_list_row.append(img_fill_row)
                image_list_col.append(image_list_row.copy())
                for xx in range(MATRIX_SIZE-len(image_list_col)):
                    image_list_col.append(img_fill_col)
                tem_img5=merge_image(image_list_col)
                img_count=img_count+1
                filename=dst_folder+"/"+prefix_str+str(img_count)+".jpg"
                cv2.imwrite(filename,tem_img5)
            break
            video.release()
    print ("totally {:d} pics,save {:d} pics".format(tem_c-1,img_count))


def merge_image(img_list):
    tem_list=[]
    img_list=add_text(img_list)
    for x in range(MATRIX_SIZE):
        tem_img2=cv2.hconcat(img_list[x])
        tem_list.append(tem_img2)
    tem_img3=cv2.vconcat(tem_list)
    return tem_img3

def add_text(img_ls):
    for xx in range(len(img_ls)):
        for yy in range(len(img_ls)):
            img_ls[xx][yy]=cv2.putText(img_ls[xx][yy], str(xx+1)+","+str(yy+1), (0, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            #各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    return img_ls

def color_img(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--i', type=str, default=None, help='Path to a list file.')
    parser.add_argument('--o', type=str, default=None, help='Path to a dataset folder.')

    parser.add_argument('--fq', type=int, default=15, help='Path to a dataset folder.')

    parser.add_argument('--matrixs', type=int, default=15, help='Path to a dataset folder.')
    parser.add_argument('--hw', type=int, default=250, help='Path to a dataset folder.')
    
    args = parser.parse_args()

    #VIDEO_PATH=r"F:\SZU-Prj\jiaonang\2019_310_capsule_prj\testdata\碧水雅居\视频数据\2019-0303\碧水雅居-Y2132_Y2128_20190303_114807.mp4"
    VIDEO_PATH=args.i
    #EXTRACT_FOLDER = './testdata/extract_folder/' # 
    EXTRACT_FOLDER =args.o
    EXTRACT_FREQUENCY =args.fq

    #MATRIX_SIZE=args.matrixsTRIX_SIZE=args.matrixs
    #Small_H_W=250 #
    #Small_H_W=args.hw
    #prefix_str="capsule22_" 
    prefix_str=VIDEO_PATH.split("\\")[-1][0:7]+"_"


    #img_fill_col = img_fill()
    #img_fill_row = np.zeros((Small_H_W,Small_H_W,3),dtype=np.uint8)
    matrix_image(VIDEO_PATH,EXTRACT_FOLDER,EXTRACT_FREQUENCY)

#python matrix_detect.py --i F:\SZU-Prj\jiaonang\2019_310_capsule_prj\testdata\Y2141-2138-20190303_135924.mp4 --o F:\SZU-Prj\jiaonang\2019_310_capsule_prj\testdata\extract_folder
#python matrix_detect.py --i F:/SZU-Prj/jiaonang/2019_310_capsule_prj/testdata/Y2141-2138-20190303_135924.mp4 --o F:/SZU-Prj/jiaonang/2019_310_capsule_prj/testdata/extract_folder
#if cv2.waitKey(1) & 0xFF == ord('q'):