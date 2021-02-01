import numpy as np       
import xlrd
import matplotlib.pyplot as plt
import pandas as pd

class excel_read:          #Construct class and import Excel data
    def __init__(self, excel_path='results_of_different_classifiers.xlsx',encoding='utf-8',index=0):

      self.my_data=xlrd.open_workbook(excel_path)  
      self.table=self.my_data.sheets()[index]     
      self.rows=self.table.nrows  



    def get_data(self):
        rr=[]
        for i in range(self.rows):
            col=self.table.row_values(i)  
            
            rr.append(col)
        
        return rr
    
def main():
    my_a=excel_read().get_data()
    x=my_a[0][1:]
    y=my_a[1:5]
    for i in range(0,4):
        y[i]=y[i][1:]
    
    y1=np.transpose(np.asarray(y))
    df = pd.DataFrame(data=y1,
                      index=x,
                      columns=['SVM','Random forest','AdaBoost','K nearest neighbors']
                     )
    df.plot(kind='bar',fontsize=10,stacked=False,figsize=(12, 8),rot=0)  
    for a,b in enumerate(df["SVM"]):
        plt.text(a-0.18,b+0.06,'%.3f'% b,ha="center",va="center",fontsize=14,rotation=90)
    for a,b in enumerate(df["Random forest"]):
        plt.text(a-0.06,b+0.06,'%.3f'% b,ha="center",va="center",fontsize=14,rotation=90)
    for a,b in enumerate(df["AdaBoost"]):
        plt.text(a+0.07,b+0.06,'%.3f'% b,ha="center",va="center",fontsize=14,rotation=90)
    for a,b in enumerate(df["K nearest neighbors"]):
        plt.text(a+0.19,b+0.06,'%.3f'% b,ha="center",va="center",fontsize=14,rotation=90)
    plt.ylim(ymax=1.13)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=14)

if __name__=='__main__':
    main()