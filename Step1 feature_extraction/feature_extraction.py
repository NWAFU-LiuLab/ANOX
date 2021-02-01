#This program can extract 1673 features based on five feature extraction methods (FRE,AADP,EEDP, and KSB). 
#The output results are two txt files (x_1673.txt and y.txt) and automatically saved in the current folder.
#The input file is anti_protein_positive_negative.txt
#The output file is used for feature selection in the next step
import re
import numpy as np
import os

def average(matrixSum, seqLen):            #A function used in extracting DPC,EEDP and KSB
    # average the summary of rows
    matrix_array = np.array(matrixSum)
    matrix_array = np.divide(matrix_array, seqLen)
    matrix_array_shp = np.shape(matrix_array)
    matrix_average = [(np.reshape(matrix_array, (matrix_array_shp[0] * matrix_array_shp[1], )))]
    return matrix_average



def preHandleColumns(PSSM,STEP,PART,ID):   #A function used in extracting DPC,EEDP and KSB
    '''
    if STEP=k, we calculate the relation betweem one residue and the kth residue afterward.
    '''
    '''
    if PART=0, we calculate the left part of PSSM.
    if PART=1, we calculate the right part of PSSM.
    '''
    '''
    if ID=0, we product the residue-pair.
    if ID=1, we minus the residue-pair.
    '''
    '''
    if KEY=1, we divide each element by the sum of elements in its column.
    if KEY=0, we don't perform the above process.
    '''
    if PART==0:
        PSSM=PSSM[:,1:21]
    elif PART==1:
        PSSM=PSSM[:, 21:]
    PSSM=PSSM.astype(float)
    matrix_final = [ [0.0] * 20 ] * 20
    matrix_final=np.array(matrix_final)
    seq_cn=np.shape(PSSM)[0]

    if ID==0:
        for i in range(20):
            for j in range(20):
                for k in range(seq_cn - STEP):
                    matrix_final[i][j]+=(PSSM[k][i]*PSSM[k+STEP][j])

    elif ID==1:
        for i in range(20):
            for j in range(20):
                for k in range(seq_cn - STEP):
                    matrix_final[i][j] += ((PSSM[k][i]-PSSM[k+STEP][j]) * (PSSM[k][i]-PSSM[k+STEP][j])/4.0)
    return matrix_final



def dpc_pssm(input_matrix):   #A function to get DPC
    PART = 0
    STEP = 1
    ID = 0
    KEY = 0
    matrix_final = preHandleColumns(input_matrix, STEP, PART, ID)
    seq_cn = float(np.shape(input_matrix)[0])
    dpc_pssm_vector = average(matrix_final, seq_cn-STEP)
    return dpc_pssm_vector[0]



def eedp(input_matrix):   #A function to get EEDP
    STEP = 2
    PART = 0
    ID = 1
    KEY = 0
    seq_cn = float(np.shape(input_matrix)[0])
    matrix_final = preHandleColumns(input_matrix, STEP, PART, ID)
    eedp_vector = average(matrix_final, seq_cn-STEP)
    return eedp_vector[0]



def k_separated_bigrams_pssm(input_matrix): #A function to get KSB
    PART=1
    ID=0
    KEY=0
    STEP = 1
    matrix_final = preHandleColumns(input_matrix, STEP, PART, ID)
    seq_cn = float(np.shape(input_matrix)[0])
    k_separated_bigrams_pssm_vector=average(matrix_final,10000.0)
    return k_separated_bigrams_pssm_vector[0]



def fun1(pssm_file):     #A function to get AADP,FRE,EEDP, and KSB
    pssm=[]
    for line in pssm_file:
        line1=re.findall(r"[\-|0-9]+",line)
        del line1[41:]
        pssm.append(line1)
    del pssm[0:3]
    del pssm[-6:]
    
    pssm_array_origin = np.array(pssm,dtype=np.float32)
    #The PSSM matrix is obtained
    pssm_array=pssm_array_origin[:,1:21]
    #Get V_PSSM
    F_pssm=np.mean(pssm_array,axis=0)
    
    bf={'A':0.082,'R':0.052,'N':0.043,'D':0.058,'C':0.017,'Q':0.039,'E':0.069,
    'G':0.073,'H':0.023,'I':0.057,'L':0.091,'K':0.062,'M':0.018,'F':0.040,
    'P':0.045,'S':0.059,'T':0.054,'W':0.014,'Y':0.035,'V':0.071}
    order={'0':'A','1':'R','2':'N','3':'D','4':'C','5':'Q','6':'E','7':'G',
       '8':'H','9':'I','10':'L','11':'K','12':'M','13':'F','14':'P',
       '15':'S','16':'T','17':'W','18':'Y','19':'V'}
    bf_array = np.array(list(bf.values()),dtype=np.float32)
    M_frequency=np.multiply(pssm_array,bf_array)#The exponential part of the frequency matrix is obtained
    S1_location_array=np.argmax(M_frequency,axis=1)#Find out the number position corresponding to the maximum number of each line of the frequency matrix
    S1_location= [str(i) for i in list(S1_location_array)]#Convert the number type in S1 to the character type to match the dictionary
    S1_list=[order[i] if i in order else i for i in S1_location]#Replace the numeric position with a specific amino acid residue
    S1=''.join(S1_list)#Get S1
    F1_gram=[]
    S1_len=len(S1)
    for n1 in order.values():
        F1_gram.append(S1.count(n1)/S1_len/21)#Get V_gram_one
    F2_gram=[]
    S1_len=len(S1)-1
    for n1 in order.values():
        for n2 in order.values():
            F2_gram.append(S1.count(n1+n2)*20/S1_len/21)#Get V_gram_two
    
    #Get DPC
    dpc_pssm_vector=dpc_pssm(pssm_array_origin)
    #Get EEDP
    eedp_vector=eedp(pssm_array_origin)
    #Get KSB
    k_separated_bigrams_pssm_vector=k_separated_bigrams_pssm(pssm_array_origin)
    #Combine V_ DPC and V_PSSM into V_AADP. Return four feature extraction methods(FRE,AADP,EEDP,KSB)
    return F1_gram+F2_gram+list(F_pssm)+list(dpc_pssm_vector)+list(eedp_vector)+list(k_separated_bigrams_pssm_vector)



def fun2(horiz_file):     #A function to get PRED based on the secondary structure sequence
    horiz=''
    Posi_h=0
    Posi_e=0
    Posi_c=0
    Posi_r=0
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
        Count_EHE=horiz4.count('H')#Remove the first and last one, and the number of remaining sequence H is the number of EHE
        F_frequency_EHE=Count_EHE/(len_horiz-2)           
    #Get V_H, V_C, V_E, V_mH, V_mE, V_βαβ
    return [FH_local,FC_local,FE_local,F_Max_H,F_Max_E,F_frequency_EHE]



def fun3(ss2_file):     #A function to get PRED based on the structure-probability matrix
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
    #Get V_global_str, V_local_str
    return list(F_pro)



def main():
    #Section 1: Import 1805 sequences
    print('Import data...')
    label_vector=[]
    train_samples=open('./Data/anti_protein_positive_negative.txt','r')
    for line in train_samples:
        if line.startswith('>'):
            label_vector.append(line.strip()[-1])
        else:
            sequence=line.strip()
    train_samples.close()
    
    
    #Section 2: Feature extraction
    print('Feature extraction starts. Please wait about 15 minutes...')
    #(1) Feature extraction based on fun1 (AADP,FRE,EEDP,KSB)
    print('Extracting AADP,FRE,EEDP,KSB')
    feature_matrix=[]
    path = "./Data/PSI-BLAST_profile" 
    files= os.listdir(path)  #Get all the file names under the folder
    files.sort(key= lambda x:int(x[1:-5]))  #Sort by path name
    '''
    #According to the path name, 1805 sequences are imported into the function 
    one by one to obtain the feature vectors. 
    Each sequence can obtain 1640 feature vectors 
    based on four feature extraction methods(AADP,FRE,EEDP,KSB)
    '''
    for file in files:
        feature_vector=[]
        position = path+'\\'+ file
        with open(position, "r",encoding='utf-8') as pssm_file:
            feature_vector.extend(fun1(pssm_file)) #Call fun1 to get 1640 features
            feature_matrix.append(feature_vector)
        pssm_file.close()
        print('Sequence '+file[1:-5]+' has finished. (1805 sequences in total)')
    feature_array = np.array(feature_matrix,dtype=np.float32)
    
    
    #(2) Feature extraction based on fun2 (PRED)
    #The secondary structure sequence is used in this section
    print("Extracting PRED")
    feature_matrix2=[]
    path2= "./Data/Structure_probability_matrix"
    files2= os.listdir(path2)
    files2.sort(key= lambda x:int(x[6:-6]))
    for file2 in files2:
        feature_vector2=[]
        position2 = path2+'\\'+ file2
        with open(position2, "r",encoding='utf-8') as horiz_file:
            feature_vector2.extend(fun2(horiz_file)) #Call fun2 to get 6 features
            feature_matrix2.append(feature_vector2)
        horiz_file.close()
    feature_array2 = np.array(feature_matrix2,dtype=np.float32)
    
    
    #(3) Feature extraction based on fun3 (PRED)
    #The structure-probability matrix is used in this section
    feature_matrix3=[]
    path3= "./Data/Secondary_structure_sequence"
    files3= os.listdir(path3)
    files3.sort(key= lambda x:int(x[6:-4]))
    for file3 in files3:
        feature_vector3=[]
        position3 = path3+'\\'+ file3
        with open(position3, "r",encoding='utf-8') as ss2_file:
            feature_vector3.extend(fun3(ss2_file)) #Call fun2 to get 27 features
            feature_matrix3.append(feature_vector3)
        ss2_file.close()
    feature_array3 = np.array(feature_matrix3,dtype=np.float32)
    
        
    ##Section 3: Summarize all the feature vectors
    feature_array_all= np.hstack((feature_array,feature_array2,feature_array3))
  
        
    #Section 4: Save the result of feature extraction in the current folder
    #The saved format is TXT, and the results can be used for feature selection
    np.savetxt("x_1673.txt", feature_array_all) #Save x
    with open("y.txt","w") as f:                #Save y
        f.writelines(label_vector)
    print('Completed.')
    print("The results of feature extraction have been saved in the current folder. The file name is x_1673.txt and y.txt")
    

    
    
if __name__=='__main__':
    main()

