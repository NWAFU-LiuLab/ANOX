import numpy as np       
import xlrd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
class excel_read:          #Construct class and import Excel data
    def __init__(self, excel_path='Pr_roc.xlsx',encoding='utf-8',index=0):

      self.my_data=xlrd.open_workbook(excel_path)  
      self.table=self.my_data.sheets()[index]     
      self.rows=self.table.nrows  



    def get_data(self):
        rr=[]
        for i in range(self.rows):
            col=self.table.row_values(i)  
            
            rr.append(col)
        
        return rr
    
class excel_read2:          #Construct class and import Excel data
    def __init__(self, excel_path='Validation_methods.xlsx',encoding='utf-8',index=0):

      self.my_data=xlrd.open_workbook(excel_path)  
      self.table=self.my_data.sheets()[index]     
      self.rows=self.table.nrows  



    def get_data(self):
        rr=[]
        for i in range(self.rows):
            col=self.table.row_values(i)  
            
            rr.append(col)
        
        return rr

def roc_figure():                #A function that plot Fig.2D
    my_a=excel_read().get_data()
    del my_a[0]
    dataset = np.array(my_a,dtype=np.float32)

    plt.text(-0.05,1.1,'D',fontsize=14,fontweight='bold')
    y_test1=dataset[:,0]
    predictions_test1=dataset[:,1]
    fpr1, tpr1, thresholds =roc_curve(y_test1, predictions_test1,pos_label=1)
    roc_auc1 = auc(fpr1, tpr1)


    y_test2=dataset[:,0]
    predictions_test2=dataset[:,2]
    fpr2, tpr2, thresholds =roc_curve(y_test2, predictions_test2,pos_label=1)
    roc_auc2 = auc(fpr2, tpr2)


    plt.title('ROC Curve',fontsize=14,y=-0.27)
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=14)
    lw = 2
    plt.plot(fpr1, tpr1, color='darkorange',
             lw=lw, label='ANOX (area = %0.3f)' % roc_auc1) 
    plt.plot(fpr2, tpr2, color='green',
             lw=lw, label='AOPs-SVM (area = %0.3f)' % roc_auc2) 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.legend(loc="lower right")
    plt.show()
    
    
    
    
def pr_figure():                #A function that plot Fig.2E
    my_a=excel_read().get_data()
    del my_a[0]
    dataset = np.array(my_a,dtype=np.float32)


    y_test1=dataset[:,0]
    predictions_test1=dataset[:,1]
    precision1,recall1,thresholds1=precision_recall_curve(y_test1, predictions_test1, pos_label= 1, sample_weight=None)
    a1=average_precision_score(y_test1, predictions_test1, average='macro', pos_label=1, sample_weight=None)


    y_test2=dataset[:,0]
    predictions_test2=dataset[:,2]
    precision2,recall2,thresholds2=precision_recall_curve(y_test2, predictions_test2, pos_label= 1, sample_weight=None)
    a2=average_precision_score(y_test2, predictions_test2, average='macro', pos_label=1, sample_weight=None)

    plt.text(-0.05,1.1,'E',fontsize=14,fontweight='bold')
    plt.title('Precision-Recall Curve',fontsize=14,y=-0.27)
    plt.xlabel('Recall',fontsize=14)
    plt.ylabel('Precision',fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=14)
    lw = 2
    plt.plot(recall1, precision1, color='darkorange',
             lw=lw, label='ANOX (area = %0.3f)' %a1) 
    plt.plot(recall2, precision2, color='green',
             lw=lw, label='AOPs-SVM (area = %0.3f)' %a2) 
    plt.plot([1, 0], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.legend(loc="lower left")
    plt.show()
    
    


def density_figure():                #A function that plot Fig.2F
    my_a=excel_read().get_data()
    del my_a[0]
    positive_set=[]
    negative_set=[]
    for i in my_a:
        if i[0]==1:
            positive_set.append(i[1])
        else:
            negative_set.append(i[1])
    positive = np.array(positive_set,dtype=np.float32)
    negative = np.array(negative_set,dtype=np.float32)

    plt.rcParams['axes.unicode_minus'] = False      
    sns.kdeplot(positive,label='Positive' ,color='r',shade=True)
    sns.kdeplot(negative,label='Negative' ,color='dodgerblue',shade=True)
    plt.xlim([0, 1])
    plt.tick_params(labelsize=14)
    plt.text(-0.05,22.5,'F',fontsize=14,fontweight='bold')
    plt.title('Kernel density curve',fontsize=14,y=-0.27)               
    plt.xlabel('Prediction score',fontsize=14)      
    plt.ylabel('Density',fontsize=14)               
    plt.show()    


#A function that plot Fig.2A, Fig.2B, Fig.2C
def tests(title,min_number,max_number,No,ax):
    my_a=excel_read2().get_data()
    x=my_a[0][2:]
    y=my_a[min_number:max_number]
    for i in range(0,2):
        y[i]=y[i][2:]
    y1=np.transpose(np.asarray(y))


    df = pd.DataFrame(data=y1,
                      index=x,
                      columns=['AOPs-SVM','ANOX']
              )
    df.plot(kind='bar',fontsize=14,stacked=False,ax=ax,rot=0,ylim=(0,1.19))  
    plt.title(title,fontsize=14,y=-0.2)
    plt.text(-0.8,1.22,No,fontsize=14,fontweight='bold')
    for a,b in enumerate(df["AOPs-SVM"]):
        plt.text(a-0.1,b+0.02,'%.3f'% b,ha="center",va="baseline",fontsize=12,rotation=90)
    for a,b in enumerate(df["ANOX"]):
        plt.text(a+0.14,b+0.02,'%.3f'% b,ha="center",va="baseline",fontsize=12,rotation=90)


def main():
    fig = plt.figure(figsize=(16,9))
    fig.subplots_adjust(hspace=0.3)
    for i in range(1, 7):
        ax = fig.add_subplot(2, 3, i)

        if i==1:     #Plot Fig.2A
            tests(title='5-CV',min_number=1,max_number=3,No='A',ax=ax)
    
        if i==2:     #Plot Fig.2B
            tests(title='The jackknife test',min_number=3,max_number=5,No='B',ax=ax)
        
        if i==3:     #Plot Fig.2C
            tests(title='Independent test',min_number=5,max_number=7,No='C',ax=ax)
    
        if i==4:     #Plot Fig.2D
            roc_figure()    
     
        if i==5:     #Plot Fig.2E
            pr_figure()
    
        if i==6:     #Plot Fig.2F
            density_figure()
            
            
            
if __name__=='__main__':
    main()
    

        
        
        
        
        
        
    
    

