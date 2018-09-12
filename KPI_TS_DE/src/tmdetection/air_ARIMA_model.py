# coding=utf-8
import pandas as pd
from tmdetection import data_stationarity
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pylab as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA, ARIMA
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import sys as sys  
 

def data():    
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
   
    data = pd.read_csv('C:\\Users\\owner\\Desktop\\AirPassengers.csv', parse_dates=['Month'], index_col='Month',
                   date_parser=dateparse)
    data.index = pd.DatetimeIndex(data.index.values,
                               freq=data.index.inferred_freq)
    ts = data['#Passengers']
    return ts


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
    # plt.show()
    

# 自相关和偏相关图，默认阶数为31阶、该函数可以用来判断数据的平稳性
def draw_acf_pacf(ts, lags=40):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=40, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=40, ax=ax2)
    # plt.show()  
    
    
def p_q_choice(timeSer, nlags=40, alpha=.05):
    kwargs = {'nlags': nlags, 'alpha': alpha}
    acf_x, confint = acf(timeSer, **kwargs)
    acf_px, confint2 = pacf(timeSer, **kwargs)

    confint = confint - confint.mean(1)[:, None]
    confint2 = confint2 - confint2.mean(1)[:, None]

    for key1, x, y, z in zip(range(nlags), acf_x, confint[:, 0], confint[:, 1]):
        if x > y and x < z:
            q = key1
            break

    for key2, x, y, z in zip(range(nlags), acf_px, confint2[:, 0], confint[:, 1]):
        if x > y and x < z:
            p = key2
            break

    return p, q


def get_ARMA(data, winSiz):  
    ts_data = np.log(data)  
    rol_mean = ts_data.rolling(window=winSiz).mean()    
    rol_mean.dropna(inplace=True)    
    ts_data_diff_1 = rol_mean.diff(1)
    ts_data_diff_1.dropna(inplace=True)
    # 这样在此差分后的数据具有快速衰减的效果
    ts_data_diff_2 = ts_data_diff_1.diff(1)
    ts_data_diff_2.dropna(inplace=True)
    print(proper_model(ts_data_diff_2, 5))
#     draw_acf_pacf(ts_data_diff_2)
    r_stationarity = data_stationarity.stationarity_result_analyse(ts_data_diff_2)
    if r_stationarity:          
#         draw_acf_pacf(ts_data_diff_2) 
        # 2,3
        model = ARMA(ts_data_diff_2, order=(4, 3)) 
        result_arma = model.fit(disp=-1, method='css')     
        return result_arma, ts_data_diff_1, rol_mean, ts_data 
    else:
        print('数据非平稳')
        return  None  
    
    
def get_ARIMA(data, winSiz):
    ts_data = np.log(data)  
    rol_mean = ts_data.rolling(window=winSiz).mean()    
    rol_mean.dropna(inplace=True)
    model = ARIMA(rol_mean, order=(1, 2, 1)) 
    result_arima = model.fit(disp=-1, method='css')     
    return result_arima


def pre_ARIMA(data, win=12):
    result = get_ARIMA(data, win) 
    predict_ts = result.predict()
    
    ts_data = np.log(data)  
    rol_mean = ts_data.rolling(window=12).mean()    
    rol_mean.dropna(inplace=True)
   
    ts_data_diff_1 = rol_mean.diff(1)
    ts_data_diff_1.dropna(inplace=True)
    # 这样在此差分后的数据具有快速衰减的效果
    ts_data_diff_2 = ts_data_diff_1.diff(1)
    ts_data_diff_2.dropna(inplace=True) 
   
    diff_shift_ts = ts_data_diff_2.shift(1)
    diff_recover_1 = predict_ts.add(diff_shift_ts)
    rol_shift_ts = result[2].shift(1)
    diff_recover = diff_recover_1.add(rol_shift_ts)
    rol_sum = result[3].rolling(window=11).sum()
    rol_recover = diff_recover * 12 - rol_sum.shift(1)
    log_recover = np.exp(rol_recover)
    log_recover.dropna(inplace=True)
    print(log_recover)   
    

def check_resid_wd_acf_pacf_qq(model):
    resid = model.resid 
    print(stats.normaltest(resid)) 
    
    """计算D-W检验的结果,越接近于2效果就好"""   
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


def pre_ARMA(data, win=12):
    result = get_ARMA(data, win)  
    
    check_resid_wd_acf_pacf_qq(result[0])
      
    # 获取预测的结果
    predict_ts = result[0].predict()
    # ts_data_diff_1进行移动
    diff_shift_ts = result[1].shift(1) 
    diff_recover_1 = predict_ts.add(diff_shift_ts)    
    # 上述是对一次一阶差分进行还原，下面是对另一次一阶差分进行还原
    rol_shift_ts = result[2].shift(1)
    diff_recover = diff_recover_1.add(rol_shift_ts)
    rol_sum = result[3].rolling(window=11).sum()
    rol_recover = diff_recover * 12 - rol_sum.shift(1)
    log_recover = np.exp(rol_recover)
    log_recover.dropna(inplace=True)  
 
    
pre_ARMA(data(), 12)
    
