# 小波离散变换demo
# -*- coding: utf-8 -*-
import pywt

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# dwt
x = np.linspace(-5, 5, 100)
print(type(x))
y = np.sin(x)
print(y)
print(type(y))
(cA, cD) = pywt.dwt(y, 'db1')
plt.subplot(311)
plt.plot(y)
plt.subplot(312)
plt.plot(cA)
plt.show()


def get_csv(csvfile):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    # paese_dates指定日期在哪列  ;index_dates将年月日的哪个作为索引 ;date_parser将字符串转为日期
    data = pd.read_csv(csvfile)
    
    return data


data = get_csv('C:\\Users\\owner\\Desktop\\AirPassengers2.csv')
ts = data['#Passengers']

ts_log = np.log(ts)
ts_diff = ts_log.diff(1).to_frame("ts_diff")
ts_diff = ts_diff.dropna()

cA, cD = pywt.dwt(ts_diff, 'db2')
print(cA)

