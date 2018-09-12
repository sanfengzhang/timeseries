# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import pywt


# 移动平均图、加权移动平均图
def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.ewma(timeSeries, span=size)

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()


# 自相关和偏相关图，默认阶数为31阶
def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()


def calcu_stationarity(timeseries):
    # 决定起伏统计
    rolmean = pd.rolling_mean(timeseries, window=12)  # 对size个数据进行移动平均
    rolstd = pd.rolling_std(timeseries, window=12)  # 偏离原始值多少
    # 画出起伏统计
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    # 进行df测试
    print('Result of Dickry-Fuller test')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value(%s)' % key] = value
    print (dfoutput)

    
def get_hdf(truth_file):
    # 读取测试测试数据集中的数据
    truth_df = pd.read_hdf(truth_file)
    # print(truth_df["KPI ID"])
    kpi_names = truth_df['KPI ID'].values
    kpi_names = np.unique(kpi_names)
    for kpi_name in kpi_names:
        # 提取KPI_ID对应的数据集
        truth = truth_df[truth_df["KPI ID"] == kpi_name]
        return  truth

    
def get_csv(csvfile):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    # paese_dates指定日期在哪列  ;index_dates将年月日的哪个作为索引 ;date_parser将字符串转为日期
    data = pd.read_csv(csvfile, parse_dates=['Month'], index_col='Month',
                   date_parser=dateparse)
    
    return data


data = get_csv('C:\\Users\\owner\\Desktop\\AirPassengers.csv')
ts = data['#Passengers']

rol_mean = ts.rolling(12).mean().to_frame("roll_mean")

rolstd = pd.rolling_std(ts, window=12).to_frame("roll_std")

diff = ts.diff(1).to_frame("diff1")


draw_acf_pacf(diff.dropna())

diff2 = ts.diff(2).to_frame("diff2")

ewma = pd.ewma(ts, span=12).to_frame("ewma")

result = pd.concat([ts, rol_mean, rolstd, ewma, diff, diff2], axis=1)
print(result)

"""
时间序列数据一般可以分为平稳或非平稳两大类
1.1 什么是时间序列
简而言之：

对某一个或者一组变量x(t)进行观察测量，将在一系列时刻t1,t2,⋯,tn所得到的离散数字组成的序列集合，称之为时间序列。
例如: 某股票A从2015年6月1日到2016年6月1日之间各个交易日的收盘价，可以构成一个时间序列；某地每天的最高气温可以构成一个时间序列。

一些特征:
趋势：是时间序列在长时期内呈现出来的持续向上或持续向下的变动。
季节变动：是时间序列在一年内重复出现的周期性波动。它是诸如气候条件、生产条件、节假日或人们的风俗习惯等各种因素影响的结果。
循环波动：是时间序列呈现出得非固定长度的周期性变动。循环波动的周期可能会持续一段时间，但与趋势不同，
                它不是朝着单一方向的持续变动，而是涨落相同的交替波动。
不规则波动：是时间序列中除去趋势、季节变动和周期波动之后的随机波动。不规则波动通常总是夹杂在时间序列中，
                    致使时间序列产生一种波浪形或震荡式的变动。只含有随机波动的序列也称为平稳序列。


在时间序列中还有一个很重要的特征就是窗口特征：
    1.如何建立窗口 ------>实时计算中Flink可以解决这个问题
    2.窗口能解决什么样的问题------>
    3.窗口宽度大小如何选择
    
    DFT离散傅里叶变换、DWT离散小波变换、PAA滑动平均聚集近似法、SAX符号聚集近似
    这些算法都会用到窗口或者类似的概念。
    
    

"""

"""
这些关于一些常用的模型概述：

在ARMA/ARIMA这样的自回归模型中，模型对时间序列数据的平稳是有要求的，因此，
需要对数据或者数据的n阶差分进行平稳检验，而一种常见的方法就是ADF检验，即单位根检验。
检测序列数据是否平稳http://www.lizenghai.com/archives/595.html

根据原始数据对其进行平稳性检验，数据不是平稳的；此时我们可以从其n阶差分，去判断n阶差分是否
是平稳的效果

MA：移动平均模型：移动平均可以抚平短期波动，反映长期趋势或周期
WMA:加权移动平均：对于一些距离现在时间长短对其进行赋予不同的权重
EWMA:指数加权移动平均

对平稳性的系列数据该如何建立模型呢，这个时候ARIMA(自回归差分移动平均)这个模型要求时间序列是平稳的

在选择自回归差分模型的时候需要确定AR(P)中P的大小
一般又两种方式：
 1.使用偏相关函数PACF
 2.利用信息准则函数

"""

adfDiff = ts.diff(1).to_frame("adfDiff")
adfDiff = adfDiff.dropna();
adfDiff = adfDiff['adfDiff']
resultADF = adfuller(adfDiff, 1)
print(resultADF)
adfDiff.plot(color='blue', label='Original')
plt.legend(loc='best')
plt.title('Rolling Mean')
plt.show()

"""





一下是对原始时间序列进行DWT离散小波变换:
离散小波变换
"""
