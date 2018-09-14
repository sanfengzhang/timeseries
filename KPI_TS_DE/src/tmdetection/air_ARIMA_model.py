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

#===============================================================================
# 时间序列数据一般可以分为平稳或非平稳两大类
# 1.1 什么是时间序列
# 简而言之：
# 
# 对某一个或者一组变量x(t)进行观察测量，将在一系列时刻t1,t2,⋯,tn所得到的离散数字组成的序列集合，称之为时间序列。
# 例如: 某股票A从2015年6月1日到2016年6月1日之间各个交易日的收盘价，可以构成一个时间序列；某地每天的最高气温可以构成一个时间序列。
# 
# 一些特征:
# 趋势：是时间序列在长时期内呈现出来的持续向上或持续向下的变动。
# 季节变动：是时间序列在一年内重复出现的周期性波动。它是诸如气候条件、生产条件、节假日或人们的风俗习惯等各种因素影响的结果。
# 循环波动：是时间序列呈现出得非固定长度的周期性变动。循环波动的周期可能会持续一段时间，但与趋势不同，
#                 它不是朝着单一方向的持续变动，而是涨落相同的交替波动。
# 不规则波动：是时间序列中除去趋势、季节变动和周期波动之后的随机波动。不规则波动通常总是夹杂在时间序列中，
#                     致使时间序列产生一种波浪形或震荡式的变动。只含有随机波动的序列也称为平稳序列。
# 
# 
# 在时间序列中还有一个很重要的特征就是窗口特征：
#     1.如何建立窗口 ------>实时计算中Flink可以解决这个问题
#     2.窗口能解决什么样的问题------>
#     3.窗口宽度大小如何选择
#     
#     DFT离散傅里叶变换、DWT离散小波变换、PAA滑动平均聚集近似法、SAX符号聚集近似
#     这些算法都会用到窗口或者类似的概念。
# 
# 这些关于一些常用的模型概述：
# 在ARMA/ARIMA这样的自回归模型中，模型对时间序列数据的平稳是有要求的，因此，
# 需要对数据或者数据的n阶差分进行平稳检验，而一种常见的方法就是ADF检验，即单位根检验。
# 检测序列数据是否平稳http://www.lizenghai.com/archives/595.html
# 
# 根据原始数据对其进行平稳性检验，数据不是平稳的；此时我们可以从其n阶差分，去判断n阶差分是否
# 是平稳的效果
# MA：移动平均模型：移动平均可以抚平短期波动，反映长期趋势或周期
# WMA:加权移动平均：对于一些距离现在时间长短对其进行赋予不同的权重
# EWMA:指数加权移动平均
# 
# 对平稳性的系列数据该如何建立模型呢，这个时候ARIMA(自回归差分移动平均)这个模型要求时间序列是平稳的
# 在选择自回归差分模型的时候需要确定AR(P)中P的大小
# 一般又两种方式：
#  1.使用偏相关函数PACF
#  2.利用信息准则函数
#===============================================================================


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
    plt.show()
    

# 自相关和偏相关图，默认阶数为31阶、该函数可以用来判断数据的平稳性
def draw_acf_pacf(ts, lags=40):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags, ax=ax2)
    plt.show()  
    

# 文中两个选取合适p，q的方法都不怎么好、在效果是上,参数还是需要自己去调试  
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
    r_stationarity = data_stationarity.stationarity_result_analyse(ts_data_diff_2)
   
    if r_stationarity: 
        model = ARMA(ts_data_diff_2, order=(2, 3)) 
        result_arma = model.fit(disp=-1, method='css')     
        return result_arma, ts_data_diff_1, rol_mean, ts_data 
    else:
        print('数据非平稳')
        return  None


def pre_ARMA(data, win=12):
    result = get_ARMA(data, win) 
    
    #===========================================================================
    # 残差白噪声检验
    # check_resid_wd_acf_pacf_qq(result[0])    #   
    # 获取预测的结果
    #===========================================================================
    predict_ts = result[0].predict()
    # ts_data_diff_1进行移动
    diff_shift_ts = result[1].shift(1) 
    diff_recover_1 = predict_ts.add(diff_shift_ts) 
    print(diff_recover_1, predict_ts)   
    # 上述是对一次一阶差分进行还原，下面是对另一次一阶差分进行还原
    rol_shift_ts = result[2].shift(1)
    diff_recover = diff_recover_1.add(rol_shift_ts)
    rol_sum = result[3].rolling(window=11).sum()
    rol_recover = diff_recover * 12 - rol_sum.shift(1)    
    log_recover = np.exp(rol_recover)
    log_recover.dropna(inplace=True)  
    ts = data[log_recover.index] 
    print(log_recover)  
    print('ARMA RMSE: %.4f' % np.sqrt(sum((log_recover - ts) ** 2) / ts.size)) 
    return result[0]
    
    
def get_ARIMA(data, winSiz):
    ts_data = np.log(data)  
    rol_mean = ts_data.rolling(window=winSiz).mean()    
    rol_mean.dropna(inplace=True)
    ts_diff = rol_mean.diff()
    ts_diff.dropna(inplace=True)    
    model = ARIMA(ts_diff, order=(2, 1, 3)) 
    result_arima = model.fit(disp=-1, method='css')     
    return result_arima, ts_diff, rol_mean, ts_data


def pre_ARIMA(data, win=12):
    result = get_ARIMA(data, win) 
    # check_resid_wd_acf_pacf_qq(result[0])
    predict_ts = result[0].predict()
    diff_shift_ts = result[1].shift(1) 
    diff_recover = predict_ts.add(diff_shift_ts)    
    rol_shift_ts = result[2].shift(1)  
    
    rol_sum = result[3].rolling(window=11).sum()
    rol_recover = (rol_shift_ts + diff_recover) * 12 - rol_sum.shift(1)
    log_recover = np.exp(rol_recover)
    log_recover.dropna(inplace=True)
    ts = data[log_recover.index]   
    print('ARMA RMSE: %.4f' % np.sqrt(sum((log_recover - ts) ** 2) / ts.size))
  

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

#===============================================================================
#===============================================================================
# data = data()
# train_size = int(len(data) * 0.67)   
# air_ARMA = pre_ARMA(data[0:train_size], 12)
# print(air_ARMA.predict('1957-01-01','1958-01-01'))
#===============================================================================

# pre_ARIMA(data, 12)
