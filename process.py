import pandas as pd
import numpy as np
import math
from math import pi
import copy
import matplotlib
from matplotlib.pyplot import MultipleLocator
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import colors

#global variable for index of the matrix
M_OPTIONNUM = 72
#M_OPTIONNUM = 84
N_INDEXNUM = 34

class Assess:
    '''
    according objective data of industrial and emergency events,
    compute resilience of the industrial
    input=data path of industrial chain and emergency events
    '''
    def __init__(self,resource_path):
        self.resource_df = pd.read_excel(resource_path).astype(float)
        #self.resource_df = pd.read_csv(resource_path,encoding='utf-8')
        self.indicator_lis = list(self.resource_df.columns)[1:]
        self.negative_indicator_lis=[4,21,22,26,27]#cost index

    def forward_processing(self):
        for j in self.negative_indicator_lis:
            max_x = self.resource_df.iloc[:,j].max()
            for i in range(M_OPTIONNUM):
                self.resource_df.iloc[i,j] = max_x - self.resource_df.iloc[i,j]
        '''
        for x in self.negative_indicator_lis:
            max_x = self.resource_df[x].max()
            for i in range(M_OPTIONNUM):
                self.resource_df[x].iloc[i] = max_x - self.resource_df[x].iloc[i]
                #self.resource_df[x].iloc[i] = 1 / (self.resource_df[x].iloc[i] + 1)
                #self.resource_df[x].iloc[i] = math.exp(-self.resource_df[x].iloc[i])
        '''
        print('forward_processing is finished.')

    def normalization(self):
        '''
        nonlinear normalization is performed using arctangent functions
        '''
        for j in range(1,N_INDEXNUM+1):
            max_x = self.resource_df.iloc[:,j].max()
            min_x = self.resource_df.iloc[:,j].min()
            for i in range(M_OPTIONNUM):
                if max_x == min_x:
                    self.resource_df.iloc[i,j] = 1
                else:
                    self.resource_df.iloc[i,j] = (self.resource_df.iloc[i,j] - min_x) / (max_x - min_x)
        print('normalization is finished.')
        '''
        maxindicator_lis=[]
        for j in range(N_INDEXNUM):
            maxindicator_lis.append(self.resource_df[self.indicator_lis[j]].max())#!
        for j in range(N_INDEXNUM):
            adv = math.ceil(maxindicator_lis[j] / 3)
            for i in range(M_OPTIONNUM):
                self.resource_df[self.indicator_lis[j]].iloc[i] = math.atan(1 / adv * self.resource_df[self.indicator_lis[j]].iloc[i]) * 2 / math.pi
                #self.resource_df[self.indicator_lis[j]].iloc[i] = math.atan(self.resource_df[self.indicator_lis[j]].iloc[i]) * 2 / math.pi
                #self.resource_df[self.indicator_lis[j]].iloc[i] = (math.exp(self.resource_df[self.indicator_lis[j]].iloc[i]) - 1) / (self.resource_df[self.indicator_lis[j]].iloc[i] + 1)
        print('normalization is finished.')
        '''

    def critic_weight(self):
        '''
        the variability and conflict between the indicators are solved.
        and the weights are objectively assigned according to the results.
        '''
        
        #variability
        deviation_s=[]
        deviation_x=[]
        for j in range(1,1+N_INDEXNUM):
            s = 0
            for i in range(M_OPTIONNUM):
                #s += pow(self.resource_df[self.indicator_lis[j]].iloc[i] - self.resource_df[self.indicator_lis[j]].mean(),2)
                s += pow(self.resource_df.iloc[i,j] - self.resource_df.iloc[:,j].mean(),2)
            deviation_s.append(pow(s / (M_OPTIONNUM - 1),0.5))
            deviation_x.append(pow(s,0.5))

        #conflict
        conflict_r=[]
        for j1 in range(1,1+N_INDEXNUM):
            r1 = 0
            for j2 in range(1,1+N_INDEXNUM):
                r2 = 0
                for i in range(M_OPTIONNUM):
                    #x1 = self.resource_df[self.indicator_lis[j1]].iloc[i] - self.resource_df[self.indicator_lis[j1]].mean()
                    #x2 = self.resource_df[self.indicator_lis[j2]].iloc[i] - self.resource_df[self.indicator_lis[j2]].mean()
                    x1 = self.resource_df.iloc[i,j1] - self.resource_df.iloc[:,j1].mean()
                    x2 = self.resource_df.iloc[i,j2] - self.resource_df.iloc[:,j2].mean()
                    r2 += x1 * x2
                    #r2 += (self.resource_df[self.indicator_lis[j1]].iloc[i] - self.resource_df[self.indicator_lis[j1]].mean()) * (self.resource_df[self.indicator_lis[j2]] - self.resource_df[self.indicator_lis[j2]].mean())
                if deviation_x[j1-1] != 0 and deviation_x[j2-1] != 0:
                    r1 += 1 - r2 / (deviation_x[j1-1] * deviation_x[j2-1])
                else:
                    r1 += 1 - 0
            conflict_r.append(r1)

        #weight
        weight=[]
        information=[]
        for j in range(N_INDEXNUM):
            information.append(deviation_s[j] * conflict_r[j])
        for j in range(N_INDEXNUM):
            weight.append(information[j] / sum(information))
        
        #update weighting matrix
        for j in range(1,1+N_INDEXNUM):
            for i in range(M_OPTIONNUM):
                x = weight[j-1] / sum(weight) * self.resource_df.iloc[i,j]
                self.resource_df.iloc[i,j] = x
                #self.resource_df[self.indicator_lis[j]].iloc[i] = weight[j] / sum(weight) * self.resource_df[self.indicator_lis[j]].iloc[i]
        print('critic_weight is finished.')
        return weight

    def ideal_resilience(self):
        '''
        the ideal solution for each index is calculated according to the weighting matrix.
        the resilience of the time point is measured by the distance between the scheme and the ideal ssolution of each index.
        return=the list of resilience(time)
        '''
        idealans_max=[]
        idealans_min=[]
        for j in range(1,1+N_INDEXNUM):
            idealans_max.append(self.resource_df.iloc[:,j].max())
            idealans_min.append(self.resource_df.iloc[:,j].min())

        resilience_lis=[]
        for i in range(M_OPTIONNUM):
            d_max=0
            d_min=0
            for j in range(1,1+N_INDEXNUM):
                d_max += pow(idealans_max[j-1] - self.resource_df.iloc[i,j],2)
                d_min += pow(idealans_min[j-1] - self.resource_df.iloc[i,j],2)    
            d_max = pow(d_max,0.5)
            d_min = pow(d_min,0.5)
            resilience_lis.append(d_min / (d_max + d_min))
        return resilience_lis
    
    def runmodule(self):
        self.forward_processing()
        self.normalization()
        self.resource_df.to_csv('D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\result\\dataframeCP.csv',encoding='utf-8')
        w_lis = self.critic_weight()
        self.resource_df.to_csv('D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\result\\dataframe_wcCP.csv',encoding='utf-8')
        r_lis = self.ideal_resilience()
        w_result_dic = {'weight':w_lis}
        w_df = pd.DataFrame(w_result_dic)
        w_df.to_csv('D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\result\\weightCP.csv',encoding='utf-8')
        r_result_dic = {'resilience':r_lis}
        r_df = pd.DataFrame(r_result_dic)
        r_df.to_csv('D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\result\\performanceCP.csv',encoding='utf-8')
        return w_lis,r_lis
    
    def sensitivity(self,filename):

        def r_distance(max_lis,min_lis,test_lis):
            d_max=0
            d_min=0
            for z in range(1,1+N_INDEXNUM):
                d_max += pow(max_lis[z-1] - test_lis[z-1],2)
                d_min += pow(min_lis[z-1] - test_lis[z-1],2)    
            d_max = pow(d_max,0.5)
            d_min = pow(d_min,0.5)
            return (d_min / (d_max + d_min))
        
        plus = 0.1
        minus = -0.1
        mean_lis = []
        max_lis = []
        min_lis = []
        resilience_index=[]
        amplitude_lis = []

        self.forward_processing()
        self.normalization()
        w_lis = self.critic_weight()

        for j in range(1, N_INDEXNUM+1):
            mean_lis.append(self.resource_df.iloc[:,j].mean())
            x_max = self.resource_df.iloc[:,j].max()
            x_min = self.resource_df.iloc[:,j].min()
            max_lis.append(x_max)
            min_lis.append(x_min)
            amplitude_lis.append(x_max-x_min)

        for j in range(1, 1+N_INDEXNUM):
            test_lis1 = []
            test_lis2 = []
            for k in range(1, 1+N_INDEXNUM):
                if k==j:
                    test_lis1.append(mean_lis[k-1] + amplitude_lis[k-1] * plus)
                    test_lis2.append(mean_lis[k-1] + amplitude_lis[k-1] * minus)
                else:
                    test_lis1.append(mean_lis[k-1])
                    test_lis2.append(mean_lis[k-1])

            resilience1 = r_distance(max_lis,min_lis,test_lis1)
            resilience2 = r_distance(max_lis,min_lis,test_lis2)
            resilience_index.append(abs(resilience1-resilience2))
        
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(7.2,2.75))
        upstream=[8,9,10,11,12,17,18,19,20,21,27,28,31,32]
        midstream=[13,14,15,16,22,23,24,25,26,29,30,33]
        tolstream=[0,1,2,3,4,5,6,7]
        indicator_name_lis=['C11','C12','C13','C14',\
                            'C21','C22','C23','C24',\
                            'C31','C32','C33','C34','C35','C36','C37','C38','C39',\
                            'C41','C42','C43','C44','C45','C46','C47','C48','C49','C410',\
                            'C51','C52','C53','C54',\
                            'C61','C62','C63']
        up_w=[[],[]]
        mid_w=[[],[]]
        tol_w=[[],[]]
        for i in range(N_INDEXNUM):
            if i in upstream:
                up_w[0].append(resilience_index[i])
                up_w[1].append(indicator_name_lis[i])
            elif i in midstream:
                mid_w[0].append(resilience_index[i])
                mid_w[1].append(indicator_name_lis[i])
            elif i in tolstream:
                tol_w[0].append(resilience_index[i])
                tol_w[1].append(indicator_name_lis[i])

        plt.figure(figsize=(7.2,2))
        #plt.title('sensitivity of resilience indicators' )
        
        plt.subplot(1,3,1)
        x1 = np.arange(len(tolstream))
        plt.title('(a) sensitivity of \nnational indicators')
        plt.bar(x1,tol_w[0],color='green')
        plt.xticks(x1,tol_w[1],rotation=30,fontsize=5)
        #plt.legend(bbox_to_anchor=(0.5,1.1),frameon=False,facecolor='none')

        plt.subplot(1,3,2)
        x2 = np.arange(len(upstream))
        plt.title('(b) sensitivity of \nupstream indicators')
        plt.bar(x2,up_w[0],color=(75/255,116/255,178/255))
        plt.xticks(x2,up_w[1],rotation=30,fontsize=5)
        #plt.legend(bbox_to_anchor=(0.5,1.1),frameon=False,facecolor='none')

        plt.subplot(1,3,3)
        x3 = np.arange(len(midstream))
        plt.title('(c) sensitivity of \nmidstream indicators')
        plt.bar(x3,mid_w[0],color=(219/255,39/255,46/255))
        plt.xticks(x3,mid_w[1],rotation=30,fontsize=5)
        #plt.legend(bbox_to_anchor=(0.5,1.1),frameon=False,facecolor='none')

        plt.subplots_adjust(wspace=0.3)
        plt.savefig(filename,dpi=600,bbox_inches='tight',transparent=True)


class Pic:
    '''
    plot
    '''
    def __init__(self,event_path,r_lis,w_lis):
        self.event_df = pd.read_excel(event_path)
        self.resilience = r_lis
        self.x_time = np.linspace(1,M_OPTIONNUM * 2,M_OPTIONNUM)
        self.name_x = list(self.event_df['时间'])
        self.x_index = np.linspace(0,N_INDEXNUM - 1,N_INDEXNUM)
        self.weight = w_lis
        self.indicator_lis=['C11','C12','C13','C14',\
                            'C21','C22','C23','C24',\
                            'C31','C32','C33','C34','C35','C36','C37','C38','C39',\
                            'C41','C42','C43','C44','C45','C46','C47','C48','C49','C410',\
                            'C51','C52','C53','C54',\
                            'C61','C62','C63']

    def line_chart_reselience(self,filename):
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        event_y = self.event_df['C14']
        y = {'Resilience Value':self.resilience,'Newly Confirmed Cases':event_y}#!
        first_y = list(y.keys())[0]
        second_y = list(y.keys())[1]
        plt.figure(figsize=(7.2,4.11))
        plt.plot(self.x_time,y.get(first_y),label=first_y,color=(75/255,116/255,178/255))
        plt.ylabel(first_y)
        plt.legend(bbox_to_anchor=(0.285,1),frameon=False,facecolor='none')
        #plt.xticks(self.x_time,self.name_x,rotation=45,fontsize='x-small',fontweight='ultralight')
        temp_x=[]
        temp_name=[]
        for i in range(len(self.x_time)):
            if i % 3 == 0:
                temp_x.append(self.x_time[i])
                temp_name.append(self.name_x[i])
        plt.xticks(temp_x,temp_name,rotation=30,fontsize=10,fontweight='ultralight')
        plt.twinx()
        plt.plot(self.x_time,y.get(second_y),label=second_y,color=(219/255,39/255,46/255))
        plt.ylabel(second_y)
        #plt.legend(loc='upper center')
        plt.legend(bbox_to_anchor=(0.365,0.95),frameon=False,facecolor='none')
        #plt.title('2018-2023 Resilience curve of Chinese rare earth industry')
        plt.savefig(filename,dpi=300,bbox_inches='tight',transparent=True)
        #plt.show()

    def bar_chart_weight(self,filename):
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(7.2,2.75))
        upstream=[8,9,10,11,12,17,18,19,20,21,27,28,31,32]
        midstream=[13,14,15,16,22,23,24,25,26,29,30,33]
        tolstream=[0,1,2,3,4,5,6,7]
        up_w=[]
        mid_w=[]
        tol_w=[]
        for i in range(N_INDEXNUM):
            if i in upstream:
                up_w.append(self.weight[i])
            elif i in midstream:
                mid_w.append(self.weight[i])
            elif i in tolstream:
                tol_w.append(self.weight[i])
        plt.bar(tolstream,tol_w,color='green')
        plt.bar(upstream,up_w,color=(75/255,116/255,178/255))
        plt.bar(midstream,mid_w,color=(219/255,39/255,46/255))

        #for i, v in enumerate(self.weight):
            #plt.annotate(str(v), xy=(i, v), ha='center', va='bottom')

        plt.xticks(self.x_index,self.indicator_lis,rotation=45,fontsize='x-small',fontweight='ultralight')
        #plt.title('Weights of Different Indicators')
        plt.xlabel('Resilience Indicators')
        plt.ylabel('Weight')
        plt.savefig(filename,dpi=300,bbox_inches='tight',transparent=True)
        #plt.show()

    def spyder_chart_index(self,res_lis,wei_lis,sec_ind):
        '''
        a transverse cross-section that generates resilience at a given moment.
        '''
        categoris=[['C11','C12','C13'],\
                   ['C21','C22','C23','C24','C25','C26','C27','C28'],\
                   ['C31','C32','C33','C34','C35','C36'],\
                   ['C41','C42','C43','C44','C45'],\
                   ['C51','C52','C53','C54','C55']]
        fig = plt.figure(figsize=(12,12))
        
        if sec_ind == 22:
            #num_ind = 0
            title_name = 'radar map of different dimensions in normal stage'
        elif sec_ind == 24:
            #num_ind = 1
            title_name = 'radar map of different dimensions in descending stage'
        elif sec_ind == 25:
            #num_ind = 2
            title_name = 'radar map of different dimensions in trough stage'
        elif sec_ind == 28:
            #num_ind = 3
            title_name = 'radar map of different dimensions in recovery stage'
        elif sec_ind == 30:
            #num_ind = 4
            title_name = 'radar map of different dimensions in completion stage'
        
        position=230
        ind_num=-1
        plt.title(title_name)
        for num_ind in range(5):
            len_num = len(categoris[num_ind])
            x = [n * 2 * pi / len_num for n in range(len_num)]
            x = np.concatenate((x,[x[0]]))

            position += 1
            ax = plt.subplot(position,polar=True)

            plt.xticks(x[:-1],categoris[num_ind],color='gray',size=8)
            plt.yticks([0.02,0.04,0.06],['0.02','0.04','0.06'],color='gray',size=7)
            plt.ylim(0,0.07)

            y=[]
            for z in categoris[num_ind]:
                ind_num += 1
                y.append(res_lis[sec_ind] * wei_lis[ind_num])
            y = np.concatenate((y,[y[0]]))
            ax.plot(x,y,'o-',linewidth=2)#连线
            ax.fill(x,y,color=(75/255,116/255,178/255),alpha=0.3)#填充

        plt.tight_layout()
        plt.savefig('D:\《突发事件下关键产业链韧性评估和提升策略》个人项目\photo\Figure03.png',dpi=300,bbox_inches='tight',transparent=True)
        plt.show()

    def spyder_chart_time(self,res_lis,wei_lis,picpath):
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        categoris=[['C11','C12','C13','C14'],\
                   ['C21','C22','C23','C24'],\
                   ['C31','C32','C33','C34','C35','C36','C37','C38','C39'],\
                   ['C41','C42','C43','C44','C45','C46','C47','C48','C49','C410'],\
                   ['C51','C52','C53','C54','C61','C62','C63']]
        plt.figure(figsize=(15,8))
        #plt.suptitle('radar map of different dimensions in different stages')
        
        position=230
        initial=0
        loc_lis=[16,24,25,28,43]
        clo_lis=[[2/255,38/255,62/255],\
                 [239/255,65/255,67/255],\
                 [191/255,30/255,46/255],\
                 [13/255,76/255,109/255],\
                 [115/255,186/255,214/255]]
        #title_name_lis=['a.Social capacity','b.Industrial independence',\
                        #'c.Competitive intensity','d.Agility',\
                        #'e.Adaptability']
        title_name_lis=['a.security','b.public management','c.market position','d.international trade','e.knowledge & sustainability']
        #title_name_lis=['(a)','(b)','(c)','(d)','(e)']
        legend_stage_lis=[]
        stage_name_lis=['201905-Adapt ability','202001-Resistant ability','202004-React ability','202003-Recover ability','202108-Learn ability']
        lim_max=[0.035, 0.035, 0.02, 0.025, 0.02]
        lim_min=[0.01,0.015,0.005,0.005,0.005]
        marker_lis=['x','1','.','+','*']

        for i in range(5):
            #plt.rcParams['font.sans-serif'] = ['Times New Roman']
            #plt.rcParams['axes.unicode_minus'] = False
            len_num = len(categoris[i])
            x = [n * 2 * pi / len_num for n in range(len_num)]
            x = np.concatenate((x,[x[0]]))

            position += 1
            ax = plt.subplot(position,polar=True)
            #ax.set_title=(title_name_lis[i])
            plt.title(title_name_lis[i], y=-0.2)

            plt.xticks(x[:-1],categoris[i],color='black',size=10)
            yt1 = lim_min[i]
            yt2 = lim_min[i]+(lim_max[i]-lim_min[i]/5)
            yt3 = lim_min[i]+(lim_max[i]-lim_min[i]/5*2)
            yt4 = lim_min[i]+(lim_max[i]-lim_min[i]/5*3)
            yt5 = lim_min[i]+(lim_max[i]-lim_min[i]/5*4)

            plt.yticks(color='gray',size=7)
            plt.ylim(lim_min[i],lim_max[i])
            
            for j in range(len(loc_lis)):
                y=[]
                jw = initial - 1
                for z in range(len_num):
                    jw += 1
                    y.append(res_lis[loc_lis[j]] * wei_lis[jw])
                y = np.concatenate((y,[y[0]]))
                pl,=ax.plot(x,y,marker=marker_lis[j],color=(clo_lis[j][0],clo_lis[j][1],clo_lis[j][2]),linewidth=2)
                #pl,=ax.plot(x,y,linewidth=2)
                legend_stage_lis.append(pl)
                #ax.fill(x,y,color=(clo_lis[i][0],clo_lis[i][1],clo_lis[i][2]),alpha=0.3)
            initial += len_num


        plt.tight_layout()
        plt.legend(handles=legend_stage_lis[0:5],labels=stage_name_lis,bbox_to_anchor=(2.5,0.8),frameon=False,facecolor='none')
        plt.savefig(picpath,dpi=300,bbox_inches='tight',transparent=True)
        plt.show()
       
class Compare:

    def __init__(self,resource_path,simulation_path):
        self.resource_df = pd.read_excel(resource_path)
        self.simulation_df = pd.read_excel(simulation_path)
        self.columns_name = self.resource_df.columns.values.tolist()
    
    def relative_error(self):
        dict = {}
        for j in range(1,N_INDEXNUM):
            lis = []
            for i in range(M_OPTIONNUM):
                if self.simulation_df.iloc[i,j] != 0:
                    re = (self.simulation_df.iloc[i,j] - self.resource_df.iloc[i,j]) / self.simulation_df.iloc[i,j]
                else:
                    if self.simulation_df.iloc[i,j] == self.resource_df.iloc[i,j]:
                        re = 0
                    else:
                        re = 1
                lis.append(re)
            lis.append(sum(lis) / M_OPTIONNUM)
            dict[self.columns_name[j]] = lis
        re_df = pd.DataFrame(dict)
        re_df.to_csv('D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\result\\relative_error.csv',encoding='utf-8')
        print('relative error has been computed.')


    def cv_rmse(self):
        dict = {}
        for j in range(1,N_INDEXNUM):
            lis = []
            var_lis = []
            averange_measure_data = self.resource_df.iloc[:,j].mean()
            for i in range(M_OPTIONNUM):
                e2 = pow(self.resource_df.iloc[i,j] - self.simulation_df.iloc[i,j], 2)
                lis.append(e2)
                var_lis.append(pow(self.resource_df.iloc[i,j] - averange_measure_data, 2))

            mse = sum(lis) / M_OPTIONNUM
            rmse = pow(mse, 0.5) / averange_measure_data
            var = sum(var_lis) / M_OPTIONNUM
            if var == 0:
                r_2 = 1 - mse
            else:
                r_2 = 1 - mse / var

            lis.append(rmse)
            lis.append(r_2)
            dict[self.columns_name[j]] = lis
        cv_rmse_df = pd.DataFrame(dict)
        cv_rmse_df.to_csv('D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\result\\cv_rmse.csv',encoding='utf-8')
        print('cv_rmse has been computed.')

class Situation:

    def __init__(self,path):
        self.df = pd.read_excel(path)
        self.x_time = np.linspace(1,84 * 2,84)
        self.policy = [['status-continuation','free-exporting','export-restricted'],\
                       ['status-continuation','expendable-reserve','increased-reserve'],\
                       ['status-continuation','focus-on-upstream-R&D','focus-on-midstream-R&D']]
        self.clo_lis=[[2/255,38/255,62/255],\
                    [239/255,65/255,67/255],\
                    [115/255,186/255,214/255]]
        self.policy_title = ['(a) Simulated Resilience in Different Export Policies',\
                             '(b) Simulated Resilience in Different Reserve Policies',\
                             '(c) Simulated Resilience in Different R&D Policies']

    def plot_fun(self,filename):
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(7,7))
        #position = 311
        position = 211
        k = 1

        temp_x=[]
        temp_name=[]
        for i in range(len(self.x_time)):
            if i % 3 == 0:
                temp_x.append(self.x_time[i])
                temp_name.append(self.df.iloc[i,0])
        
        for z in range(2):#3
            ax = plt.subplot(position)
            plt.title(self.policy_title[z])
            plt.plot(self.x_time,self.df.iloc[:,1],label=self.policy[z][0],color=self.clo_lis[0])
            for j in range(1,3):
                plt.plot(self.x_time,self.df.iloc[:,j+k],label=self.policy[z][j],color=self.clo_lis[j])
            
            plt.ylabel('resilience performance')
            #plt.xlabel('time')
            plt.xticks(temp_x,temp_name,rotation=30,fontsize=10,fontweight='ultralight')
            plt.legend(frameon=False,facecolor='none')

            position += 1
            k += 2
        
        plt.tight_layout()
        plt.savefig(filename,dpi=300,bbox_inches='tight',transparent=True)
        plt.show()


        
if __name__=='__main__':
    resource_path='D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\MODEL_DATA.xls'
    event_path='D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\PREASURE_DATA.xls'
    assess = Assess(resource_path)
    weight_lis,resilience_lis = assess.runmodule()
    
    picture = Pic(event_path,resilience_lis,weight_lis)
    picture.line_chart_reselience('D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\result\\FIG031101.png')
    picture.bar_chart_weight('D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\result\\FIG031102.png')
    picture.spyder_chart_time(resilience_lis,weight_lis,'D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\result\\FIG031104.png')
    assess.sensitivity('D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\result\\FIG031103SEN.png')
        

    compare_path='D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\SITUATION_MERGE1.xls'
    
    '''
    #simulation_path='D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\COMPARE_DATA6.xls'
    #simulation_path='D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\SITUATION_DATA3_UP.xls'
    #assess1 = Assess(simulation_path)
    weight_lis,resilience_lis = assess1.runmodule()
    
    compare = Compare(resource_path,simulation_path)
    compare.relative_error()
    compare.cv_rmse()
    '''
    com_sit = Situation(compare_path)
    com_sit.plot_fun('D:\\《突发事件下关键产业链韧性评估和提升策略》个人项目\\2018-2024数据\\result\\FIG031105.png')
    