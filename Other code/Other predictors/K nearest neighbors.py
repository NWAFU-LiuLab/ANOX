# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import auc
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

def main():
    feature_array_all=np.loadtxt('x_1170.txt',dtype=np.float32)
    f = open("y.txt", "rb")
    label_vector= f.read().decode()
    label_vector=list(label_vector)
    f.close()
    label_vector = np.array(label_vector,dtype=np.float32)

    #The independent testing dataset is taken out and cannot participate in 5-CV
    X_trainset,X_testset,y_trainset,y_testset=train_test_split(feature_array_all,label_vector,test_size=0.2,random_state=0,stratify=label_vector)

    X=X_trainset
    y=y_trainset
    skf = StratifiedKFold(n_splits=5, random_state=2, shuffle=True)
    skf.get_n_splits(X, y)

    ACC_sum=0
    roc_auc_sum=0
    Sn_sum=0
    Sp_sum=0
    F1_sum=0
    MCC_sum=0
    cnt=1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = KNeighborsClassifier()
        clf=clf.fit(X_train,y_train)
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
        print('Times=%s'%cnt)
        print('k nearest neighbors ACC:%s'%ACC)
        print('k nearest neighbors AUC:%s'%roc_auc)
        print('k nearest neighbors Sn:%s'%Sn)
        print('k nearest neighbors Sp:%s'%Sp)
        print('k nearest neighbors F1:%s'%F1)
        print('k nearest neighbors MCC:%s'%MCC)
        ACC_sum+=ACC
        roc_auc_sum+=roc_auc
        Sn_sum+=Sn
        Sp_sum+=Sp
        F1_sum+=F1
        MCC_sum+=MCC
        cnt+=1

    ACC=ACC_sum/5
    roc_auc=roc_auc_sum/5
    Sn=Sn_sum/5
    Sp=Sp_sum/5
    F1=F1_sum/5
    MCC=MCC_sum/5
    print('')
    print('5-Fold cross validation_Conclusion')
    print('k nearest neighbors ACC:%s'%ACC)
    print('k nearest neighbors AUC:%s'%roc_auc)
    print('k nearest neighbors Sn:%s'%Sn)
    print('k nearest neighbors Sp:%s'%Sp)
    print('k nearest neighbors F1:%s'%F1)
    print('k nearest neighbors MCC:%s'%MCC)

if __name__=='__main__':
    main()
