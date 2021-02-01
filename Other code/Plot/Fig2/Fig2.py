import xlrd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
class excel_read:          #Construct class and import Excel data
    def __init__(self, excel_path='The_MRMD_score.xlsx',encoding='utf-8',index=0):

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
    def __init__(self, excel_path='Top_ten_features.xlsx',encoding='utf-8',index=0):

      self.my_data=xlrd.open_workbook(excel_path)  
      self.table=self.my_data.sheets()[index]     
      self.rows=self.table.nrows  

    def get_data(self):
        rr=[]
        for i in range(self.rows):
            col=self.table.row_values(i)  
            
            rr.append(col)
        
        return rr



def Fig_a():                #A function that plot Fig.2A
    my_a=excel_read().get_data()
    x=[i for i in range(0,1673)]
    y=my_a[1:]
    for i in range(0,1673):
        y[i]=y[i][2]

    plt.bar(x, y,width=4,color='orange')
    plt.xlim(xmin=0,xmax=1749)

    plt.vlines(420, 0, 1.05, colors = "k", linestyles = "dotted")
    plt.vlines(840, 0, 1.05, colors = "k", linestyles = "dotted")
    plt.vlines(1240, 0, 1.05, colors = "k", linestyles = "dotted")
    plt.vlines(1640, 0, 1.05, colors = "k", linestyles = "dotted")
    plt.text(210,0.9,'FRE',fontsize=20,horizontalalignment='center' )
    plt.text(630,0.9,'AADP',fontsize=20,horizontalalignment='center' )
    plt.text(1030,0.9,'EEDP',fontsize=20,horizontalalignment='center' )
    plt.text(1430,0.9,'KSB',fontsize=20,horizontalalignment='center' )
    plt.text(1700,0.9,'PRED',fontsize=20,horizontalalignment='center' )
    plt.title('A',fontsize=14,x=0,fontweight='bold')
    plt.tick_params(labelsize=14)
    plt.ylabel('The MRMD Score',fontsize=14)
    plt.xlabel('Feature identifier',fontsize=14)    
    
    

def Fig_b():                #A function that plot Fig.2B
    my_a=excel_read2().get_data()
    y=my_a[1:]
    x=my_a[1:]
    for i in range(0,10):
        y[i]=y[i][1]
        x[i]=x[i][0] 
    plt.barh(x, y, height=0.6)
    plt.xlim(xmax=1.19)
    for a,b in zip(x,y):
        plt.text(b+0.08,a,'%.3f'% b,ha="center",va="center",fontsize=14)
    plt.title('B',fontsize=14,x=0,fontweight='bold')
    plt.ylabel('Feature name',fontsize=14)
    plt.xlabel('The MRMD Score',fontsize=14)
    plt.tick_params(labelsize=14)


    
def main():
    fig = plt.figure(figsize=(14,9))

    #Plot Fig.2A
    grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.3)
    ax=plt.subplot(grid[0, 0:])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    Fig_a()

    #Plot Fig.2B
    plt.subplot(grid[1, 0])
    Fig_b()

    #Plot Fig.2C
    axes=plt.subplot(grid[1, 1]);
    box_plot = pd.read_excel('Box_plot.xlsx')
    sns.boxplot(x='Feature name',y='Data',hue='category',
                data=box_plot,orient='v',ax=axes,fliersize=3)
    plt.ylim(ymax=1.49)
    plt.legend(loc="upper right")
    plt.title('C',fontsize=14,x=0,fontweight='bold')
    plt.ylabel('Feature extraction data',fontsize=14)
    plt.xlabel('Feature name',fontsize=14)
    plt.tick_params(labelsize=14)



if __name__=='__main__':
    main()
