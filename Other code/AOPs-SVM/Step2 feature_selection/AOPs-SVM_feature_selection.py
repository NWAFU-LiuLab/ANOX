# -*- coding: utf-8 -*-
#This program can select the best feature combination by MRMD score. 
#The input file is x_473.txt and y.txt obtained in the process of feature extraction.
#The output file is x_189.txt, which can be used for the next step of model construction and test
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def main():
    #Part 1: import x, y after feature extraction
    RVi=[]
    DVi=[]
    MRMD_scorei=[]
    feature_array_all=np.loadtxt('x_473.txt',dtype=np.float32)
    f = open("y.txt", "rb")
    label_vector= f.read().decode()
    label_vector=list(label_vector)
    f.close()


    #Part 2: the independent test dataset is taken out and cannot participate in feature selection
    label_vector = np.array(label_vector,dtype=int)
    feature_array_copy=feature_array_all[:,0:]
    X_trainset,X_testset,y_trainset,y_testset=train_test_split(feature_array_all,label_vector,test_size=0.2,random_state=0,stratify=label_vector)

    feature_array_all=X_trainset
    label_vector=y_trainset





    #Part 3: Using MRMD score to analyze features in descending order
    for i in range(0,feature_array_all.shape[1]):
        DV=0.0
        RV=np.corrcoef(feature_array_all[:,i], label_vector)
        RVi.append(RV[0,1])
        for j in range(0,feature_array_all.shape[1]):
            EDij=np.linalg.norm(feature_array_all[:,i]-feature_array_all[:,j])
            DV+=EDij
        DVi.append(DV/feature_array_all.shape[1])
    for k in range(0,feature_array_all.shape[1]):
        MRMD_scorei.append(RVi[k]+ DVi[k])


    MRMD_scorei = np.array(MRMD_scorei,dtype=np.float32)
    rank=np.argsort(MRMD_scorei)
    rank=rank[::-1]  
    feature_array_all=feature_array_all[...,rank]
    feature_array_copy=feature_array_copy[...,rank]


    #Part 4: Eliminate redundant data through the MRMD score and find the feature combination of the highest AUC index
    X_train_original,X_test_original,y_train,y_test=train_test_split(feature_array_all,label_vector,test_size=0.2,random_state=3,stratify=label_vector)

    conclusion={}
    for cnt in range(1,feature_array_all.shape[1]+1):
        clf=RandomForestClassifier(random_state=0)
        X_train=X_train_original[...,0:cnt]
        X_test=X_test_original[...,0:cnt]
        clf=clf.fit(X_train,y_train)
        predict_y_test = clf.predict(X_test)
    
        prob_predict_y_test = clf.predict_proba(X_test)
        predictions_test = prob_predict_y_test[:, 1]
        y_validation=np.array(y_test,dtype=int)
        F1=metrics.f1_score(y_validation, np.array(predict_y_test, int))
        conclusion[cnt] = F1
        print('Random forest has been completed %s times. 473 times in total'%cnt)

    for key, value in conclusion.items():
        if value == max(conclusion.values()):
            max_key = key


    
    #Part 6: Result output
    np.savetxt("x_189.txt", feature_array_copy[...,0:max_key]) 
    print("The results of feature selection have been saved in the current folder. The file name is x_189.txt")    
    print('Number of feature selection: %s'%max_key)
    
if __name__=='__main__':
    main()




    
    
    



