import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd
import shutil
import os
import json
from sklearn import svm
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

from PIL import Image

# load train data 
# param: csv arra
#表格映射

def map_out(input,mapt,gt_file):

    gt_label=jsonread(gt_file)
    
    assert len(input)==len(mapt)
    out={"1":[],"-1":[]}
    for x in range(len(input)):
        if input[x]==1:
            out["1"].append(mapt[x])
        else:
            out["-1"].append(mapt[x])
    out["-1"].sort()
    out["1"].sort()
    true_mapt_label=[]
    for x in range(len(input)):
        if mapt[x] in gt_label["1"]:
            true_mapt_label.append(1)
        else:
            true_mapt_label.append(-1)
    
    return out,true_mapt_label

def load_train_data(csv_path,start=1,end=None,process_type="std"):
    dframe_train=pd.read_csv(csv_path,sep=",",header=0,index_col=False)
    if bool(end):
        X_data=dframe_train.iloc[:,start:end].values
    else:
        X_data=dframe_train.iloc[:,start:].values

    X_map_table=dframe_train.iloc[:,0].values

    if process_type=="std":
        ss = StandardScaler()
        ss.fit(X_data)
        X_data= ss.transform(X_data)

    return X_data,X_map_table

def load_train_data_combine(csv_path,feat_select,process_type="std"):
    dframe_train=pd.read_csv(csv_path,sep=",",header=0,index_col=False)
    feat_list=[]
    for k in feat_select:
        one_list=[x for x in dframe_train.columns if k in x]
        one_arr=dframe_train.loc[:len(dframe_train),one_list].values
        feat_list.append(one_arr)
    X_data=np.hstack(feat_list)
    X_map_table=dframe_train.iloc[:,0].values

    if process_type=="std":
        ss = StandardScaler()
        ss.fit(X_data)
        X_data= ss.transform(X_data)

    return X_data,X_map_table

#计算正确率并写入文件中
#关于正确率 召回率 https://blog.csdn.net/qq_29007291/article/details/86080456
# python两个列表获取交集，并集，差集 https://www.cnblogs.com/jiaoxiaohui/p/10429526.html
#假设:
#一组样本，个数为，正例有P个，负例有N个，

#算法结果:
#判断为正例的正例有TP个，判断为负例的正例有FN个(假的负例）P=TP+FN
##判断为负例的负例为TN个，判断为正例的负例有FP个（假的正例）N=TN+FP

#指标计算:
#正例 精确度（Precision）P=所有判断为正例的例子中，真正为正例的所占的比例=TP/(TP+FP)
#负例 精确度（Precision）P=所有判断为负例例的例子中，真正为负例的所占的比例=TN/(TN+FN)
#准确率（Accuracy）A=判断正确的例子的比例=（TP+TN）/（P+N）
#召回率（Recall）R=所有正例中，被判断为正例的比例=TP/P
#漏警概率=1-Recall，正例判断错误的概率，漏掉的正例所占比率
#虚警率=1-Precision，错误判断为正例的概率，虚假正例所占的比率

# def json_roc_save(json_path,roc_table):
#     dict_tmp={}
#     fpath_tmp=file_dir_name(path)
#     for x in fpath_tmp:
#         fn_tmp=os.listdir(os.path.join(path,x))
#         dict_tmp[x]=fn_tmp
#         dict_tmp[x+"_len"]=len(fn_tmp)
    
#     f=open(file_gt,"w+")
#     f.write(str(dict_tmp).replace("'","\""))
#     f.close()
#     return dict_tmp


def jsonread(gt_file):
    gt_tmp2={"1":[],"-1":[]}
    f=open(gt_file,"r")
    gt_tmp=json.load(f)
    f.close()
    #total_tmp1=len(gt_tmp["1"])+len(gt_tmp["-1"])
    for x in gt_tmp["-1"]:
        gt_tmp2["-1"].append(int(x.replace(".jpg","")))
    for xx in gt_tmp["1"]:
        gt_tmp2["1"].append(int(xx.replace(".jpg","")))
    return gt_tmp2

def evalu_acc(gt_file,res_map):
    gt_tmp=jsonread(gt_file)

    total_tmp1=len(gt_tmp["1"])+len(gt_tmp["-1"])
    total_tmp2=len(res_map["1"])+len(res_map["-1"])
    assert total_tmp1==total_tmp2

    res_acc={"precision_b1":None,"precision":None,"acc":None,"recall_b1":None,
            "recall":None,"false_alert_b1":None,"miss_alert_b1":None,
            "false_alert":None,"miss_alert":None,"intersect_1":None,
            "intersect_b1":None}
    #交集 1intersect_1
    intersect_1=list(set(gt_tmp["1"]).intersection(set(res_map["1"])))

    #交集 -1
    intersect_b1=list(set(gt_tmp["-1"]).intersection(set(res_map["-1"])))
    #FP=len(res_map["-1"])-len(intersect_b1)
    res_acc["intersect_1"]=len(intersect_1)
    res_acc["intersect_b1"]=len(intersect_b1)

    if len(res_map["1"])==0:
        res_acc["precision"]=len(res_map["1"])
    else:
        res_acc["precision"]=np.round(len(intersect_1)/len(res_map["1"]),decimals=4)

    if len(res_map["-1"])==0:
        res_acc["precision_b1"]=len(res_map["-1"])
    else:
        res_acc["precision_b1"]=np.round(len(intersect_b1)/len(res_map["-1"]),decimals=4)
    
    res_acc["acc"]=np.round((len(intersect_1)+len(intersect_b1))/(len(gt_tmp["1"])+len(gt_tmp["-1"])),decimals=4)
    
    #负例子
    res_acc["recall_b1"]=np.round(len(intersect_b1)/len(gt_tmp["-1"]),decimals=4)
    res_acc["recall"]=np.round(len(intersect_1)/len(gt_tmp["1"]),decimals=4)

    res_acc["miss_alert_b1"]=1-res_acc["recall_b1"]
    res_acc["false_alert_b1"]=1-res_acc["precision_b1"]

    #例子
    
    res_acc["miss_alert"]=1-res_acc["recall"]
    res_acc["false_alert"]=1-res_acc["precision"]
    
    return res_acc

def outliner_detect(X_train,map_table,X_test=None,file_name=None,
                    gt_file=None,algorithm_M=None,params_feat=None):
    #需要预测
    # fit the model
    y_pred_train=None
    if algorithm_M["Alogrithm"]=="iForest":
        clf = IsolationForest(max_samples=algorithm_M["max_samples"],contamination=algorithm_M["contamination"],
                                behaviour=algorithm_M["behaviour"],max_features=algorithm_M["max_features"])
        clf.fit(X_train)
        y_pred_train = clf.predict(X_train)

        score_decision=clf.decision_function(X_train)
        res,true_mapt_label=map_out(y_pred_train,map_table,gt_file)
        roc_table=roc_curve(true_mapt_label,score_decision,pos_label=1)
        #clf.predict_proba()
    if algorithm_M["Alogrithm"]=="oneClass_SVM":
        clf = svm.OneClassSVM(nu=algorithm_M["nu"], kernel=algorithm_M["kernel"], gamma=algorithm_M["gamma"])
        clf.fit(X_train)
        y_pred_train = clf.predict(X_train)

        score_decision=clf.decision_function(X_train)
        res,true_mapt_label=map_out(y_pred_train,map_table,gt_file)
        roc_table=roc_curve(true_mapt_label,score_decision,pos_label=1)

    if algorithm_M["Alogrithm"]=="EllipticEnvelop":
        clf = EllipticEnvelope(contamination=algorithm_M["contamination"],support_fraction=0.7)
        clf.fit(X_train)
        y_pred_train = clf.predict(X_train)

        score_decision=clf.decision_function(X_train)
        res,true_mapt_label=map_out(y_pred_train,map_table,gt_file)
        roc_table=roc_curve(true_mapt_label,score_decision,pos_label=1)
        
    if algorithm_M["Alogrithm"]=="LocalOutlierFactor":
        clf = LocalOutlierFactor(n_neighbors=algorithm_M["n_neighbors"],contamination=algorithm_M["contamination"])
        #clf.fit(X_train)
        y_pred_train = clf.fit_predict(X_train)

        score_decision=clf.negative_outlier_factor_
        res,true_mapt_label=map_out(y_pred_train,map_table,gt_file)
        roc_table=roc_curve(true_mapt_label,score_decision,pos_label=1)

    if algorithm_M["Alogrithm"]=="k-means++":
        clf = KMeans(init=algorithm_M["Alogrithm"], n_clusters=algorithm_M["n_clusters"], n_init=algorithm_M["n_init"])
        clf.fit(X_train)
        y_pred_tem = clf.predict(X_train)
        y_pred_train=cluster_ouliners(y_pred_tem,thresh=algorithm_M["thresh"])

        score_decision=[]#clf.score(X_train)
        res,true_mapt_label=map_out(y_pred_train,map_table,gt_file)
        roc_table=[np.array([1]),np.array([1]),np.array([1])]

    if algorithm_M["Alogrithm"]=="DBSCAN":
        clf = DBSCAN(eps=algorithm_M["eps"], min_samples=algorithm_M["min_samples"]).fit(X_train)#eps=algorithm_M["eps"], min_samples=algorithm_M["min_samples"]
        y_pred_tem=clf.labels_
        #y_pred_tem = clf.predict(X_train)

        y_pred_train=cluster_ouliners(y_pred_tem,thresh=algorithm_M["thresh"])
        score_decision=[]#clf.decision_function(X_train)
        res,true_mapt_label=map_out(y_pred_train,map_table,gt_file)
        roc_table=[np.array([1]),np.array([1]),np.array([1])]
            
    if algorithm_M["Alogrithm"]=="AggCluster":
        clf =AgglomerativeClustering(n_clusters=algorithm_M["n_clusters"], linkage=algorithm_M["linkage"]).fit(X_train)
        y_pred_tem=clf.labels_
        #y_pred_tem = clf.predict(X_train)
        y_pred_train=cluster_ouliners(y_pred_tem,thresh=algorithm_M["thresh"])
        score_decision=[]#clf.decision_function(X_train)
        res,true_mapt_label=map_out(y_pred_train,map_table,gt_file)
        roc_table=[np.array([1]),np.array([1]),np.array([1])]

    assert bool(algorithm_M)

    #y_pred_test = clf.predict(X_test)
    #y_pred_outliers = clf.predict(X_outliers)

    thresh_tem=""
    if algorithm_M["thresh"]:
        thresh_tem=algorithm_M["thresh"]

    if bool(file_name):
        precision=evalu_acc(gt_file,res)
        f=open(file_name,"w+")
        f.write("param settings:"+str(clf)+"\n")
        f.write("cluster_thresh:"+str(thresh_tem)+"\n")
        f.write("feature select:"+str(params_feat)+"\n")
        
        f.write("precision: " +str(precision)+"\n")
        f.write("all_predict:\n")
        f.write("sum:"+str(len(y_pred_train))+"\n")
        f.write("1 class :"+str(np.sum(y_pred_train==1))+"\n")
        f.write(str(res["1"])+"\n")
        f.write("-1 class :"+str(np.sum(y_pred_train==-1))+"\n")
        f.write(str(res["-1"])+"\n\n")
        f.write("score : "+str(score_decision)+"\n")
        f.close()
    return res,precision,roc_table
    #不需要预测

def mv_result(in_dir,out_dir,res_csv):
    out_file1=os.path.join(out_dir,"1")
    out_file2=os.path.join(out_dir,"-1")

    if os.path.exists(out_file1):
        shutil.rmtree(out_file1)
    os.mkdir(out_file1)
    if os.path.exists(out_file2):
        shutil.rmtree(out_file2)
    os.mkdir(out_file2)
    for x in res_csv["1"]:
        in_file1=os.path.join(in_dir,str(x)+".jpg")
        out_file1_tem=os.path.join(out_file1,str(x)+".jpg")
        img=Image.open(in_file1)
        im = img.resize((int(img.size[0]*0.2), int(img.size[1]*0.2)), Image.ANTIALIAS)
        im.save(out_file1_tem)
        #shutil.copy(in_file1,out_file1)
    for xx in res_csv["-1"]:
        in_file2=os.path.join(in_dir,str(xx)+".jpg")
        out_file2_tem=os.path.join(out_file2,str(xx)+".jpg")
        img=Image.open(in_file2)
        im = img.resize((int(img.size[0]*0.2), int(img.size[1]*0.2)), Image.ANTIALIAS)
        im.save(out_file2_tem)
        #shutil.copy(in_file2,out_file2)

def cluster_result(pred,thresh=2):
    res_dict={}
    out={}
    target=np.unique(pred)
    tem_li=[]
    for x in target:
        res_dict[str(x)]=np.where(pred==x)[0]
        tem_li.append(np.where(pred==x)[0].size)
    tem2_li=sorted(tem_li)
    res_list=[]
    for k in range(len(tem_li)):
        res_list.append(res_dict[str(tem_li.index(tem2_li[k]))])
    
    list_tem1=[]
    for j in range(len(res_list)-thresh):
        list_tem1.append(res_list[j+thresh])
    out_1=np.hstack(list_tem1).reshape(-1).tolist()
    out_1.sort()

    #异常点
    list_tem2=[]
    for i in range(thresh):
        list_tem2.append(res_list[i])
    out_2=np.hstack(list_tem2).reshape(-1).tolist()
    out_2.sort()
    out["-1"]=out_2
    out["1"]=out_1
    #res_dict,res_list
    return out


def cluster_ouliners(pred_in,thresh=0.6):
    target=np.unique(pred_in).tolist()
    target.sort()
    ada_thresh=int(np.round(thresh*len(target)))
    if ada_thresh==0:
        ada_thresh=1
    tem_list=[]
    for x in target:
        tem_list.append(np.sum(pred_in==x))
    tem2_list=sorted(tem_list)
    predo=pred_in.copy()
    for k in range(len(tem_list)):
        if k <ada_thresh:
            predo[predo==tem_list.index(tem2_list[k])]=-1
        else:
            predo[predo==tem_list.index(tem2_list[k])]=1
    return predo.reshape(-1)

#创建文件
def create_Dir(algorithm_In=None,feat_select=None,root_path="./"):
    
    if feat_select:
        pre_dir="_"
        dir_tem=pre_dir.join(feat_select)
        out_res_path1=os.path.join(root_path,algorithm_In["Alogrithm"],dir_tem)
        if os.path.exists(out_res_path1):
            shutil.rmtree(out_res_path1)
        os.makedirs(out_res_path1)
        fileN=os.path.join(out_res_path1,"result.txt")
        return fileN,out_res_path1

    else:
        out_res_path1=os.path.join(root_path,algorithm_In["Alogrithm"])
        if os.path.exists(out_res_path1):
            shutil.rmtree(out_res_path1)
        os.makedirs(out_res_path1)
        
        return out_res_path1
    # out_res_path2=os.path.join(root_path,pre_dir,algorithm_In["Alogrithm"])
    # if os.path.exists(out_res_path2):
    #     shutil.rmtree(out_res_path2)
    # os.makedirs(out_res_path2)

    #file_name=os.path.join(out_res_path,"result.txt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--alogrithm",type=str,default="a7")
    parser.add_argument("--feat_select",type=str,default="f1")
    parser.add_argument("--cut_l",type=int,default=320)
    parser.add_argument("--cut_r",type=int,default=1720)  
    args = parser.parse_args()
    #public params

    # data1
    gt_file=r"G:\Linux-Proj\capuse_detection\data1-gt\all-label\gt_label.json"
    path=r"G:\Linux-Proj\capuse_detection\data1-gt\all_features_1.csv"
    img_dir=r"G:\Linux-Proj\capuse_detection\data1-gt\all-2"
    out_dir=r"G:\Linux-Proj\capuse_detection\experiment1\data1_res2"
    
    # gt_file=r"G:\Linux-Proj\capuse_detection\data4-gt\all-label\gt_label.json"
    # path=r"G:\Linux-Proj\capuse_detection\data4-gt\all_features_4.csv"
    # img_dir=r"G:\Linux-Proj\capuse_detection\data4-gt\all-2"
    # out_dir=r"G:\Linux-Proj\capuse_detection\experiment1\data4_res1"
    
    
    #res_filename="res.txt"
    feat_combine1=["lbp","gab","glcm","hog","ori"]
    feat_combine2=["lbp","gab","glcm","hog"]
    feat_combine3=["lbp"]
    feat_combine4=["gab"]
    feat_combine5=["glcm"]
    feat_combine6=["hog"]
    feat_combine7=["ori"]

    #生成训练数据与映射表格
    #x_data,xmap_tab=load_train_data(path,start=start,end=end)

    cluster_thresh=0.6
    #method params
    iForest_M={"Alogrithm":"iForest","contamination":0.2,"behaviour":"new","max_features":1.0,"max_samples":100,"thresh":None}
    oneClass_SVM_M={"Alogrithm":"oneClass_SVM","nu":0.2,"kernel":"rbf","gamma":0.2,"thresh":None}
    Gaussion_D_M={"Alogrithm":"EllipticEnvelop","contamination":0.2,"thresh":None}
    LocalOutlierFactor_M={"Alogrithm":"LocalOutlierFactor","n_neighbors":35,"contamination":0.2,"thresh":None}
    #cluster method
    Kmeans_M={"Alogrithm":"k-means++","n_clusters":5,"n_init":10,"thresh":cluster_thresh}
    DBSCAN_M={"Alogrithm":"DBSCAN","eps":3,"min_samples":5,"thresh":cluster_thresh}
    AgglomerativeClustering_M={"Alogrithm":"AggCluster","n_clusters":10,"linkage":'ward',"thresh":cluster_thresh}


    #algorithm_M=[iForest_M,oneClass_SVM_M,Gaussion_D_M ,LocalOutlierFactor_M,Kmeans_M,DBSCAN_M,AgglomerativeClustering_M]
    algorithm_M=[iForest_M,Gaussion_D_M]#,LocalOutlierFactor_M]oneClass_SVM_M,
    
    feat_select=[feat_combine1,feat_combine2,feat_combine3,feat_combine4,feat_combine5,feat_combine6,feat_combine7]
    #algorithm_M=Kmeans_M
    res_roc_json={}
    for algo in algorithm_M:
        res_dict={}
        if algo["Alogrithm"]=="EllipticEnvelop":
            res_dict={"Gaussion-D":["precision_0","precision","acc","recall_0","recall"]}
        else:
            res_dict={algo["Alogrithm"]:["precision_0","precision","acc","recall_0","recall"]}
        al_path=create_Dir(algorithm_In=algo,feat_select=None,root_path=out_dir)
        for feat_s in feat_select:
            str_tem="_"
            feat_path,out_res_path=create_Dir(algorithm_In=algo,feat_select=feat_s,root_path=out_dir)
            #载入数据
            x_data,xmap_tab=load_train_data_combine(path,feat_s)

            res,precison,roc_table=outliner_detect(x_data,xmap_tab,file_name=feat_path,gt_file=gt_file,
                            algorithm_M=algo,params_feat=feat_s)
            
            res_roc_json[algo["Alogrithm"]+"_"+str_tem.join(feat_s)]=[list(roc_table[0]),list(roc_table[1]),list(roc_table[2])]

            res_dict[str_tem.join(feat_s)]=[precison["precision_b1"],precison["precision"],precison["acc"]
                                            ,precison["recall_b1"],precison["recall"]]
            #mv_result(img_dir,out_res_path,res)
        if algo["Alogrithm"]=="EllipticEnvelop":
            algo_csv_path=os.path.join(al_path,"Gaussion-D"+".csv")
        else:
            algo_csv_path=os.path.join(al_path,algo["Alogrithm"]+".csv")
            

        res_dict_df = pd.DataFrame(data=res_dict)
        res_dict_df.transpose().to_csv(algo_csv_path, sep=',', header=True,index=True)
    
    json_roc_path=os.path.join(out_dir,"roc.json")

    with open(json_roc_path, 'w') as f:
         f.write(str(res_roc_json).replace("'","\""))

             
    #移动生成结果



    #异常点检测 与 结果映射
    #res_f_save=os.path.join(img_out_path,res_filename)





    


