#This program can select the best feature combination by MRMD score. 
#The input file is x_1673.txt and y.txt obtained in the process of feature extraction.
#The output file is x_1170.txt, which can be used for the next step of model construction and test
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import auc


def main():
    #Part 1: import x, y after feature extraction
    feature_array_all=np.loadtxt('x_1673.txt',dtype=np.float32)
    f = open("y.txt", "rb")
    label_vector= f.read().decode()
    label_vector=list(label_vector)
    f.close()
    label_vector = np.array(label_vector,dtype=np.float32)


    #Part 2: data normalization
    min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1, 1))
    feature_array_all= min_max_scaler.fit_transform(feature_array_all)
    feature_array_copy=feature_array_all[:,0:]
    
    
    #Part 3: the independent test dataset is taken out and cannot participate in feature selection
    X_trainset,X_testset,y_trainset,y_testset=train_test_split(feature_array_all,label_vector,test_size=0.2,random_state=0,stratify=label_vector)

    feature_array_all=X_trainset
    label_vector=y_trainset
    feature_name_1=['FRE'+str(i) for i in range(1,421)]
    feature_name_2=['AADP'+str(i) for i in range(1,421)]
    feature_name_3=['EEDP'+str(i) for i in range(1,401)]
    feature_name_4=['KSB'+str(i) for i in range(1,401)]
    feature_name_5=['PRED'+str(i) for i in range(1,34)]
    feature_name=feature_name_1+feature_name_2+feature_name_3+feature_name_4+feature_name_5



    #Part 4: Using MRMD score to analyze features in descending order
    RVi=[]
    DVi=[]
    MRMD_scorei=[]
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
    feature_array_all=feature_array_all[...,rank] #Feature ranking
    feature_array_copy=feature_array_copy[...,rank] #The feature copies are also sorted
    rank_list=list(rank)
    feature_name_rank=[feature_name[x] for x in rank_list]


    #Part 5: Eliminate redundant data through the MRMD score and find the feature combination of the highest AUC index
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
        fpr, tpr, thresholds =metrics.roc_curve(y_validation, predictions_test,pos_label=1)
        roc_auc = auc(fpr, tpr)
        conclusion[cnt] = roc_auc
        print('Random forest has been completed %s times. 1673 times in total'%cnt)

    for key, value in conclusion.items():
        if value == max(conclusion.values()):
            max_key = key

 
    #Part 6: Result output
    np.savetxt("x_1170.txt", feature_array_copy[...,0:max_key])
    print("The results of feature selection have been saved in the current folder. The file name is x_1170.txt")    
    print('Number of feature selection: %s'%max_key)
    
    
if __name__=='__main__':
    main()