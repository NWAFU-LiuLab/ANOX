#This program can extract 1673 features based on AOPs-SVM. 
#The output results are two txt files (x_473.txt and y.txt) and automatically saved in the current folder.
#The input file is anti_protein_positive_negative.txt
#The output file is used for feature selection in the next step
import re
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import auc


def fun1(pssm_file):
    pssm=[]
    for line in pssm_file:
        line1=re.findall(r"[\-|0-9]+",line)
        del line1[41:]
        pssm.append(line1)
    del pssm[0:3]
    del pssm[-6:]
    
    pssm_array_origin = np.array(pssm,dtype=np.float32)
    pssm_array=pssm_array_origin[:,1:21]
    F_pssm=np.mean(pssm_array,axis=0)
    
    bf={'A':0.082,'R':0.052,'N':0.043,'D':0.058,'C':0.017,'Q':0.039,'E':0.069,
    'G':0.073,'H':0.023,'I':0.057,'L':0.091,'K':0.062,'M':0.018,'F':0.040,
    'P':0.045,'S':0.059,'T':0.054,'W':0.014,'Y':0.035,'V':0.071}
    order={'0':'A','1':'R','2':'N','3':'D','4':'C','5':'Q','6':'E','7':'G',
       '8':'H','9':'I','10':'L','11':'K','12':'M','13':'F','14':'P',
       '15':'S','16':'T','17':'W','18':'Y','19':'V'}
    bf_array = np.array(list(bf.values()),dtype=np.float32)
    M_frequency=np.multiply(pssm_array,bf_array)
    Sc_location_array=np.argmax(M_frequency,axis=1)
    Sc_location= [str(i) for i in list(Sc_location_array)]
    Sc_list=[order[i] if i in order else i for i in Sc_location]
    Sc=''.join(Sc_list)
    F1_gram=[]
    Sc_len=len(Sc)
    for n1 in order.values():
        F1_gram.append(Sc.count(n1)/Sc_len/21)
    F2_gram=[]
    Sc_len=len(Sc)-1
    for n1 in order.values():
        for n2 in order.values():
            F2_gram.append(Sc.count(n1+n2)*20/Sc_len/21)

    return F1_gram+F2_gram+list(F_pssm)


def fun2(horiz_file):
    horiz=''
    Posi_h=0
    Posi_e=0
    Posi_c=0
    for line in horiz_file:
        if line.startswith('Pred'):
             horiz+=line.strip()[6:]
    len_horiz=len(horiz)
    for i2 in range(1,len_horiz+1):
        if horiz[i2-1]=='H':
            Posi_h+=i2
        elif horiz[i2-1]=='E':
            Posi_e+=i2  
        else:
            Posi_c+=i2
    FH_local=Posi_h/(len_horiz*(len_horiz-1))
    FE_local=Posi_e/(len_horiz*(len_horiz-1))
    FC_local=Posi_c/(len_horiz*(len_horiz-1))

    max_length={'max_length_H':0,'max_length_E':0,'max_length_C':0}
    current_line=1
    for i3 in range(0,len_horiz-1):
        if horiz[i3]==horiz[i3+1]:
            current_line+=1
        if horiz[i3]!=horiz[i3+1]:
            max_length['max_length_'+horiz[i3]]=max(max_length['max_length_'+horiz[i3]],current_line)
            current_line=1
    F_Max_E=max_length['max_length_E']/len_horiz
    F_Max_H=max_length['max_length_H']/len_horiz
    
    
    
    horiz2=horiz.replace('C','')
    if len(horiz2)<=2:
        F_frequency_EHE=0
    else:
        horiz3=horiz2[0]
        for i4 in range(1,len(horiz2)):
            if horiz2[i4]!=horiz2[i4-1]:
                horiz3+=horiz2[i4]
        horiz4=horiz3[1:-1]
        Count_EHE=horiz4.count('H')
        F_frequency_EHE=Count_EHE/(len_horiz-2)           
    return [FH_local,FC_local,FE_local,F_Max_H,F_Max_E,F_frequency_EHE]




def fun3(ss2_file):
    ss2=[]
    ss2_samples=ss2_file
    for line in ss2_samples:
        line1=re.findall(r'[\d+\.]+',line)
        ss2.append(line1)
    del ss2[0]
    del ss2[0]
    lenth=len(ss2)
    M_probability= np.array(ss2,dtype=np.float32)
    M_probability=np.delete(M_probability,0,axis = 1)
    F_pro_global=np.mean(M_probability, axis =  0)


    F_pro=F_pro_global.copy()
    local_lenth=int(lenth/8)
    c =np.array_split(M_probability,8,axis=0)
    for i in c:
        F_pro_local=np.mean(i, axis = 0)
        F_pro=np.append(F_pro, F_pro_local, axis=None)
    return list(F_pro)




def main():
    label_vector=[]
    train_samples=open('./Data/anti_protein_positive_negative.txt','r')
    for line in train_samples:
        if line.startswith('>'):
            label_vector.append(line.strip()[-1])
        else:
            sequence=line.strip()
    train_samples.close()
    
    
    
    feature_matrix=[]
    path = "./Data/PSI-BLAST_profile" 
    files= os.listdir(path)
    files.sort(key= lambda x:int(x[1:-5]))
    for file in files:
        feature_vector=[]
        position = path+'\\'+ file
        with open(position, "r",encoding='utf-8') as pssm_file:
            feature_vector.extend(fun1(pssm_file))
            feature_matrix.append(feature_vector)
        pssm_file.close()
    feature_array = np.array(feature_matrix,dtype=np.float32)
    
    
   
    feature_matrix2=[]
    path2= "./Data/Secondary_structure_sequence"
    files2= os.listdir(path2)
    files2.sort(key= lambda x:int(x[6:-6]))
    for file2 in files2:
        feature_vector2=[]
        position2 = path2+'\\'+ file2
        with open(position2, "r",encoding='utf-8') as horiz_file:
            feature_vector2.extend(fun2(horiz_file))
            feature_matrix2.append(feature_vector2)
        horiz_file.close()
    feature_array2 = np.array(feature_matrix2,dtype=np.float32)
    
    
    
    
    
    feature_matrix3=[]
    path3= "./Data/Structure_probability_matrix"
    files3= os.listdir(path3)
    files3.sort(key= lambda x:int(x[6:-4]))
    for file3 in files3:
        feature_vector3=[]
        position3 = path3+'\\'+ file3
        with open(position3, "r",encoding='utf-8') as ss2_file:
            feature_vector3.extend(fun3(ss2_file))
            feature_matrix3.append(feature_vector3)
        ss2_file.close()
    feature_array3 = np.array(feature_matrix3,dtype=np.float32)
    
    feature_array_all= np.hstack((feature_array,feature_array2,feature_array3))
  
    #Section 4: Save the result of feature extraction in the current folder
    #The saved format is TXT, and the results can be used for feature selection
    np.savetxt("x_473.txt", feature_array_all) #Save x
    with open("y.txt","w") as f:               #Save y
        f.writelines(label_vector)
    print("The results of feature extraction have been saved in the current folder. The file name is x_473.txt and y.txt")
    


        
if __name__=='__main__':
    main()

