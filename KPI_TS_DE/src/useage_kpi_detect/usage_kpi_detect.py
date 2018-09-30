import pandas as pd
from tmdetection import data_stationarity
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np 
from statsmodels.tsa.arima_model import ARMA, ARIMA
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import sys as sys   
import datetime

from statsmodels.tsa.seasonal import seasonal_decompose



def getData():
    truth_df = pd.read_hdf('D:\\kpi\\1.hdf')
    # print(truth_df["KPI ID"])
    kpi_names = truth_df['KPI ID'].values
    truth = truth_df[truth_df["KPI ID"] == kpi_names[0]]
    
    y = truth['label']    
    data= truth.drop(columns=['label', 'KPI ID'])
    data=data.drop_duplicates(['timestamp'],keep='first') 
    data['timestamp']=data['timestamp'].map(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
    ts = pd.Series(data['value'].values, index=pd.DatetimeIndex(data['timestamp']))
    return ts

# 自相关和偏相关图，默认阶数为31阶、该函数可以用来判断数据的平稳性
def draw_acf_pacf(ts, lags=40):   
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags, ax=ax2)
    plt.show()  
    
    
    
def get_acf_pacf(ts_log_diff):
    # 确定参数
    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
    # q的获取:ACF图中曲线第一次穿过上置信区间.这里q取2
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')  # lowwer置信区间
    plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')  # upper置信区间
    plt.title('Autocorrelation Function')
    # p的获取:PACF图中曲线第一次穿过上置信区间.这里p取2
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()
        
  
def proper_model(data_ts, maxLag):
    init_bic = sys.maxsize
    print(init_bic)
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(data_ts, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_ARMA.bic
            print(bic, p, q)
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARMA
                init_bic = bic
    return init_bic, init_p, init_q, init_properModel 

def check_resid_wd_acf_pacf_qq(model):
    """残差白噪声序列检验、计算D-W检验的结果,越接近于2效果就好""" 
    resid = model.resid 
    print(stats.normaltest(resid))
    print(sm.stats.durbin_watson(resid))
    
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    fig = qqplot(resid, line='q', ax=ax, fit=True)
    
    r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    data = np.c_[range(1, 41), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    print(table.set_index('lag'))    
 
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(resid, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(resid, lags=40, ax=ax2)
    plt.show() 
    
    
data=getData()
data1=np.log(data)
rol_mean = data1.rolling(window=240).mean()    
rol_mean_diff=rol_mean.diff(1).dropna()
r_stationarity = data_stationarity.stationarity_result_analyse(rol_mean_diff)

model = ARMA(rol_mean_diff, order=(0, 11)) 
result_arma = model.fit(disp=-1, method='css') 
check_resid_wd_acf_pacf_qq(result_arma)    






   
#===============================================================================
# print(r_stationarity)
# draw_acf_pacf(rol_mean_diff.to_frame("diff1"))
# print(proper_model(rol_mean_diff, 2))
#===============================================================================
