# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import auc
from sklearn import svm

def main():
    #Import x_189.txt and y.txt
    feature_array_all=np.loadtxt('x_189.txt',dtype=np.float32)
    f = open("y.txt", "rb")
    label_vector= f.read().decode()
    label_vector=list(label_vector)
    f.close()
    label_vector = np.array(label_vector,dtype=np.float32)


    #The independent testing dataset is taken out and cannot participate in model training
    X_train,X_test,y_train,y_test=train_test_split(feature_array_all,label_vector,test_size=0.2,random_state=0,stratify=label_vector)
    
    #Use svm
    clf=svm.SVC(probability=True,C=20.6913808111479,gamma=0.25118864315095824)
    clf=clf.fit(X_train,y_train)
    #Test model using independent testing dataset
    score_r=clf.score(X_test,y_test)
    predict_y_test = clf.predict(X_test)
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(0,len(y_test)):
        if int(y_test[i])==1 and int(predict_y_test[i])==1:
            TP=TP+1
        elif int(y_test[i])==1 and int(predict_y_test[i])==0:
            FN=FN+1
        elif int(y_test[i])==0 and int(predict_y_test[i])==0:
            TN=TN+1
        elif int(y_test[i])==0 and int(predict_y_test[i])==1:
            FP=FP+1
    Sn=float(TP)/(TP+FN)
    Sp=float(TN)/(TN+FP)
    ACC=float((TP+TN))/(TP+TN+FP+FN)
    prob_predict_y_test = clf.predict_proba(X_test)
    predictions_test = prob_predict_y_test[:, 1]
              
    y_validation=np.array(y_test,dtype=int)
    fpr, tpr, thresholds =metrics.roc_curve(y_validation, predictions_test,pos_label=1)
    roc_auc = auc(fpr, tpr)
    F1=metrics.f1_score(y_validation, np.array(predict_y_test, int))
    MCC=metrics.matthews_corrcoef(y_validation, np.array(predict_y_test, int))
    print('svm ACC:%s'%ACC)
    print('svm AUC:%s'%roc_auc)
    print('svm Sn:%s'%Sn)
    print('svm Sp:%s'%Sp)
    print('svm F1:%s'%F1)
    print('svm MCC:%s'%MCC) 
    
    
if __name__=='__main__':
    main()    







