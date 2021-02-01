# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
from sklearn.metrics import auc
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
def main():
    feature_array_all=np.loadtxt('x_189.txt',dtype=np.float32)
    f = open("y.txt", "rb")
    label_vector= f.read().decode()
    label_vector=list(label_vector)
    f.close()
    label_vector = np.array(label_vector,dtype=np.float32)

    loo = LeaveOneOut()
    X=feature_array_all
    y=label_vector
    predict_y_test=np.empty(0)
    predictions_test=np.empty(0)
    for train_index, test_index in loo.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf=svm.SVC(probability=True,C=20.6913808111479,gamma=0.25118864315095824)
        clf=clf.fit(X_train,y_train)
        score_r=clf.score(X_test,y_test)
        predict_y_test_single = clf.predict(X_test)
        predict_y_test=np.append(predict_y_test, predict_y_test_single, axis=None)
        prob_predict_y_test =clf.predict_proba(X_test)
        predictions_test_single=prob_predict_y_test[:, 1]
        predictions_test = np.append(predictions_test,predictions_test_single,axis=None)
        print('Sequence '+str(test_index[0]+1)+' has finished. (1805 sequences in total)')
    

    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(0,len(y)):
        if int(y[i])==1 and int(predict_y_test[i])==1:
            TP=TP+1
        elif int(y[i])==1 and int(predict_y_test[i])==0:
            FN=FN+1
        elif int(y[i])==0 and int(predict_y_test[i])==0:
            TN=TN+1
        elif int(y[i])==0 and int(predict_y_test[i])==1:
            FP=FP+1
    Sn=float(TP)/(TP+FN)
    Sp=float(TN)/(TN+FP)
    ACC=float((TP+TN))/(TP+TN+FP+FN)
    y_validation=np.array(y,dtype=int)
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
