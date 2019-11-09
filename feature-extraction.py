import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import transform 
from skimage.feature import local_binary_pattern,hog,daisy,greycomatrix, greycoprops
from skimage import data,exposure,io,measure
from skimage.color import label2rgb
from tqdm import tqdm
from sklearn.decomposition import PCA
from numba import jit
import numba
from skimage.filters import gabor_kernel,gabor
from scipy import ndimage as ndi
import glob
from PIL import Image as PImage


@jit(parallel=True)
def lbp_feature(img,n_point=24,radius=3,method='uniform',n_components=1):
    lbp = local_binary_pattern(img, n_point, radius, method)
    #n_bins = int(lbp.max() + 1)
    hist=np.histogram(lbp,bins=int(lbp.max() + 1),density=1)
    #pca = PCA(n_components=n_components,whiten=True)
    #result = pca.fit_transform(lbp)
    return hist[0].reshape(-1)#26

@jit()
def hog_feature(image,n_components=4,
                orientations=12,pixels_per_cell=16,
                cells_per_block=1):
    fd = hog(image, orientations=12,
            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
            cells_per_block=(cells_per_block, cells_per_block),
            visualize=False, multichannel=False,feature_vector=False)

    fd=fd.reshape([-1,orientations]).transpose()
    pca = PCA(n_components=n_components,whiten=True)
    newX = pca.fit_transform(fd)
    #print(pca.explained_variance_ratio_)
    return newX.reshape(-1)#12*4=48

@jit(parallel=True)
def daisy_feature(image,n_components=5):
    img=transform.resize(image,(min(image.shape),min(image.shape)))#(min(image.shape),min(image.shape))
    descs= daisy(img, step=120, radius=40, rings=2, histograms=6,
                             orientations=8, visualize=False)

    descs=descs.reshape([-1,104])#.transpose() 104=8*6*2+8                    
    pca = PCA(n_components=n_components,whiten=True)
    result = pca.fit_transform(descs)
    return result #36*5=180

def gabor_feature(image,freq_tupe=(0.1, 0.25)):
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in freq_tupe:
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return np.array(feats).reshape([-1])#32

def ori_img_feature(img,f_vector=56):
    #img_pil=np.array(PImage.open(path).convert("L").resize((224,224)))
    img_pil=img.reshape([f_vector,-1])
    pca = PCA(n_components=2,whiten=True)
    newX = pca.fit_transform(img_pil)
    return newX.reshape([-1])

@jit(parallel=True)
def glcm_feature(image):
    glcm = greycomatrix(image, [2,8,16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                        256, symmetric=True, normed=True)
    #print(len(glcm))
    arr=np.empty((0,0))
    for prop in {'contrast', 'dissimilarity',
             'homogeneity', 'correlation', 'ASM'}:#, 'energy'
        temp = greycoprops(glcm, prop)
        temp=np.array(temp).reshape(-1)
        arr=np.append(arr,temp)
    entropy_tem=[]
    for k in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            entropy_tem.append(measure.shannon_entropy(glcm[:,:,k,j]))
    entropy_feat=np.array(entropy_tem)
    arr=np.append(arr,entropy_feat)
    return arr#.reshape([-1]) #4*3*6=72
   
#@jit(parallel=True)
def main(fname,header=True):
    result=""
    #header=True
    fn_list=glob.glob(fname)
    for x in tqdm(fn_list):
        #fn_tem=os.path.join(path_dir,str(x+1)+".jpg")
        image=np.array(PImage.open(x).convert("L").resize((224,224),resample=PImage.BILINEAR))
        
        #feature extract

        gab_f=np.round(gabor_feature(image),decimals=8)
        lbp_f=np.round(lbp_feature(image),decimals=8)
        hog_f=np.round(hog_feature(image),decimals=8)
        glcm_f=np.round(glcm_feature(image),decimals=8)
        ori_f=np.round(ori_img_feature(image),decimals=8)

        if header:
            firstline="idx,"
            for xi in range(gab_f.size):
                firstline=firstline+"gab_"+str(xi+1)+","
            for xi in range(lbp_f.size):
                firstline=firstline+"lbp_"+str(xi+1)+","
            for xi in range(hog_f.size):
                firstline=firstline+"hog_"+str(xi+1)+","
            for xi in range(glcm_f.size):
                firstline=firstline+"glcm_"+str(xi+1)+","
            for xi in range(ori_f.size):
                firstline=firstline+"ori_"+str(xi+1)+","
            firstline=firstline[:-1]+"\n"
            header=False
            result=result+firstline

        #print(desc_day.reshape([-1]))
        str_tem1=str(gab_f.tolist())[1:-1].replace(", ",",")#.replace("\n","")#.replace(",,","")
        str_tem2=str(lbp_f.tolist())[1:-1].replace(", ",",")
        str_tem3=str(hog_f.tolist())[1:-1].replace(", ",",")
        str_tem4=str(glcm_f.tolist())[1:-1].replace(", ",",")#.replace("\n","")#.replace(",,",",")
        str_tem5= str(ori_f.tolist())[1:-1].replace(", ",",")#.replace("\n","")#.replace(",,",",")
        idx=x[x.rindex("\\")+1:x.index(".")]
        feature_cont=str_tem1+","+str_tem2+","+str_tem3+","+str_tem4+","+str_tem5+"\n"
        result=result+str(idx)+","+feature_cont#.replace(",,",",")

    return result

if __name__ == "__main__":
    # METHOD = 'uniform'
    #path=r"F:\SZU-Prj\jiaonang\2019_310_capsule_prj\detect-damage\data\extract-2\236.jpg"
    #img=io.imread(path)[0]*1.0#,as_gray=True)
    #path_dir=r"F:\SZU-Prj\jiaonang\2019_310_capsule_prj\detect-damage\data\extract-2\*.jpg"
    path_dir=r"G:\Linux-Proj\capuse_detection\data4-gt\all-2\*.jpg"
    #path_dir=r"F:\SZU-Prj\jiaonang\2019_310_capsule_prj\detect-damage\data\oneClassSVM\test\2\*.jpg"
    #image=np.array(PImage.open(path).convert("L"))
    #glcm_feature(image)
    #hist1=lbp_feature(image,16,2,'uniform')
    #hog_f=hog_feature(image)
    #img_pil=np.array(PImage.open(path_dir).convert("L").resize((224,224)))
    #res=gabor_feat(img_pil)
    #print(np.array(res).shape)
    #********
    #fname=os.listdir(path_dir)
    out_features=os.path.join(os.path.dirname(path_dir),"all_features_4.csv")

    f=open(out_features,"w+")
    res1=main(path_dir)
    f.write(res1)
    f.close()
    # print("done!")
    # print(hog_f)
    # print(hist1[0])
    #desc_day=daisy_feature(image)
    #print(desc_day)
#


#GLCM https://blog.csdn.net/nima1994/article/details/81135158




#******************daisy feature**********
    #image = data.camera()
    # image=transform.resize(image,(min(image.shape),min(image.shape)))#(min(image.shape),min(image.shape))
    # descs, descs_img = daisy(image, step=120, radius=40, rings=2, histograms=6,
    #                          orientations=8, visualize=True)

    # descs=descs.reshape([-1,104])#.transpose()
    # pca = PCA(n_components=15,whiten=False)
    # newX = pca.fit_transform(descs)
    # print(newX.shape)
    # s=pca.explained_variance_ratio_
    # print(s,sum(s))
    # fig, ax = plt.subplots()
    # ax.axis('off')
    # ax.imshow(descs_img)
    # descs_num = descs.shape[0] * descs.shape[1]
    # ax.set_title('%i DAISY descriptors extracted:' % descs_num)
    # plt.show()




#************************* HOG TEST******************************
#https://blog.csdn.net/zhanghenan123/article/details/80853523
#PCA: http://www.cnblogs.com/pinard/p/6243025.html
#https://blog.csdn.net/qq_29422251/article/details/51638087

    #image = data.astronaut()

    # fd, hog_image = hog(image, orientations=12, pixels_per_cell=(16, 16),
    #                    cells_per_block=(1, 1), visualize=True, multichannel=False,feature_vector=True)
    # fd=fd.reshape([-1,12]).transpose()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    # pca = PCA(n_components=4)
    # new1 = pca.fit_transform(fd)

    # print(pca.explained_variance_ratio_)
    # print(new1)

    # pca2 = PCA(n_components=3)

    # ax1.axis('off')
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Input image')

    # #Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()
#************************* LBP TEST******************************
    # hist1=lbp_feature(image,24,3,'uniform')
    # lbp = local_binary_pattern(image, 24, 3,'uniform')#ror
    # print(lbp.shape)
    # print(hist1[0].shape)
    # #plt.hist(hist1[0],hist1[1])
    # n, bins, patches = plt.hist(lbp.ravel(), bins=int(lbp.max() + 1), density=1,edgecolor='None',facecolor='red')
    # #n, bins, patches = plt.hist(image.ravel(), bins=100, density=1,edgecolor='None',facecolor='red')  
    
    # plt.plot(.5*(hist1[1][1:]+hist1[1][:-1]),hist1[0])
    # plt.show()
    #print(len(n),n[0].shape)
    #print(bins)

    #image =data.camera()*1.0
    
