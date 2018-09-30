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
    print(type(ts))
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    print(ts)
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
