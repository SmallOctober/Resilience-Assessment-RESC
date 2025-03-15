import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
import statsmodels.api as sm


# 创建一个函数来检查数据的平稳性
def test_stationarity(timeseries):
    # 执行Dickey-Fuller测试
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

# 对原始数据进行纯随机性和平稳性检验
def original_data_test(df,col,save_path1,save_path2):
    
    stationary_lis1 = []
    stationary_lis2 = []
    stationary_df = {}

    # LB检验的原假设是不相关，即该数据是纯随机的，如果拒绝原假设，说明该数据有一定的规律，可以建模。
    result = lb_test(df.iloc[:,col], return_df=True,lags=5)
    print(result)
    
    # 检查原始数据的平稳性
    test_stationarity(df.iloc[:,col])
    # 一阶差分
    stationary_lis1 = df.iloc[:,col] - df.iloc[:,col].shift(1)
    stationary_df['first_difference'] = stationary_lis1
    save1dic = {'1':stationary_lis1[1:]}
    save1df = pd.DataFrame(save1dic)
    save1df.to_csv(save_path1)
    # 检查一阶差分后的数据的平稳性
    test_stationarity(stationary_df['first_difference'].dropna())

    # 进行二阶差分
    stationary_lis2 = stationary_df['first_difference'] - stationary_df['first_difference'].shift(1)
    stationary_df['second_difference'] = stationary_lis2
    save2dic = {'2':stationary_lis2[2:]}
    save2df = pd.DataFrame(save2dic)
    save2df.to_csv(save_path2)
    # 检查二阶差分后的数据的平稳性
    test_stationarity(stationary_df['second_difference'].dropna())
    
    # 可视化原始数据和差分后的数据 判断参数d
    plt.figure(figsize=(12, 6))
    plt.plot(df.iloc[:,col], label='Original')
    plt.plot(stationary_df['first_difference'], label='1st Order Difference')
    plt.plot(stationary_df['second_difference'], label='2nd Order Difference')
    plt.legend(loc='best')
    plt.title('Original and Differenced Time Series')
    plt.show()

def acf_pacf(df,col):
    # 根据ACF和PACF判断参数p q
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df.iloc[:,col].values.squeeze(), lags=20, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df.iloc[:,col], lags=20, ax=ax2)
    plt.tight_layout()
    plt.show()

def arima(ori_df,col,p,d,q,pre_path):

    arma_mod = ARIMA(ori_df.iloc[:,col], order=(p, d, q)).fit()
    print(arma_mod.summary())
    predict_sunspots = arma_mod.predict(start=1, end=84, dynamic=False)
    
    dic={}
    dic[str(p)+','+str(d)+','+str(q)] = predict_sunspots
    pre_df = pd.DataFrame(dic)
    pre_df.to_csv(pre_path)
    
    plt.figure(figsize=(10,4))
    plt.plot(ori_df.iloc[:,0],ori_df.iloc[:,col],label='actual')
    plt.plot(predict_sunspots.index,predict_sunspots,label='predict')
    plt.legend(['actual','predict'])
    plt.xlabel('time(month)')
    plt.ylabel('simulation value')
    plt.show()

#original_df = pd.read_csv("D:\\relationship_data.csv", encoding='utf-8')
original_df = pd.read_excel('D:\\od1.xls')
index = 2
sta_path1 = 'D:\\sta1_df21.csv'
sta_path2 = 'D:\\sta2_df21.csv'

#step1
original_data_test(original_df,index,sta_path1,sta_path2)


#step2
sta_df = pd.read_csv(sta_path2)
sta_col = 1
acf_pacf(sta_df,sta_col)

#acf_pacf(original_df,index)


#step3
p = 12
d = 1
q = 6
predict_path = 'D:\\pre_df.csv'
arima(original_df,index,p,d,q,predict_path)

