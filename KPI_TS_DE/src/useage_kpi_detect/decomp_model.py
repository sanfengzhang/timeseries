#-*- coding:utf-8 -*-
'''
周期性时间序列预测
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from useage_kpi_detect import test_stationarity
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from datetime import timedelta
import datetime as datetime




class ModelDecomp(object):
    def __init__(self, file, test_size=1440):
        self.ts = self.read_data(file)
        self.test_size = test_size
        self.train_size = len(self.ts) - self.test_size
        self.train = self.ts[:len(self.ts)-test_size]
        self.train = self._diff_smooth(self.train)
        # draw_ts(self.train)
        self.test = self.ts[-test_size:]

    def read_data(self, f):
        truth_df = pd.read_hdf(f)
        kpi_names = truth_df['KPI ID'].values
        truth = truth_df[truth_df["KPI ID"] == kpi_names[0]]
        data= truth.drop(columns=['label', 'KPI ID'])
        data['timestamp']=data['timestamp'].map(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))    
          
        ts = pd.Series(data['value'].values, index=data['timestamp'])
        return ts

    def _diff_smooth(self, ts):
        dif = ts.diff().dropna()
        
        return dif

    def decomp(self, freq):
        '''
        对时间序列进行分解
        :param freq: 周期
        '''
        decomposition = seasonal_decompose(self.train, freq=freq, two_sided=False)
        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.residual = decomposition.resid
        decomposition.plot()
        plt.show()

        d = self.residual.describe()
        delta = d['75%'] - d['25%']

        self.low_error, self.high_error = (d['25%'] - 1 * delta, d['75%'] + 1 * delta)

    def trend_model(self, order):
        '''
        为分解出来的趋势数据单独建模
        '''
        self.trend.dropna(inplace=True)
        self.trend_model = ARIMA(self.trend, order).fit(disp=-1, method='css')

        return self.trend_model

    def add_season(self):
        '''
        为预测出的趋势数据添加周期数据和残差数据
        '''
        self.train_season = self.seasonal
        values = []
        low_conf_values = []
        high_conf_values = []

        for i, t in enumerate(self.pred_time_index):
            trend_part = self.trend_pred[i]

            # 相同时间的数据均值
            season_part = self.train_season[
                self.train_season.index.time == t.time()
                ].mean()

            # 趋势+周期+误差界限
            predict = trend_part + season_part
            low_bound = trend_part + season_part + self.low_error
            high_bound = trend_part + season_part + self.high_error

            values.append(predict)
            low_conf_values.append(low_bound)
            high_conf_values.append(high_bound)

        self.final_pred = pd.Series(values, index=self.pred_time_index, name='predict')
        self.low_conf = pd.Series(low_conf_values, index=self.pred_time_index, name='low_conf')
        self.high_conf = pd.Series(high_conf_values, index=self.pred_time_index, name='high_conf')

    def predict_new(self):
        '''
        预测新数据
        '''
        #续接train，生成长度为n的时间索引，赋给预测序列
        n = self.test_size
        self.pred_time_index= pd.date_range(start=self.train.index[-1], periods=n+1, freq='1min')[1:]
        self.trend_pred= self.trend_model.forecast(n)[0]

        self.add_season()


def evaluate(filename):
    md = ModelDecomp(file=filename, test_size=1440)
    md.decomp(freq=1440)
    md.trend_model(order=(1, 1, 3))
    md.predict_new()
    pred = md.final_pred
    test = md.test

    plt.subplot(211)
    plt.plot(md.ts)
    plt.title(filename.split('.')[0])
    plt.subplot(212)
    pred.plot(color='salmon', label='Predict')
    test.plot(color='steelblue', label='Original')
    md.low_conf.plot(color='grey', label='low')
    md.high_conf.plot(color='grey', label='high')

    plt.legend(loc='best')
    plt.title('RMSE: %.4f' % np.sqrt(sum((pred.values - test.values) ** 2) / test.size))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    filename = 'D:/kpi/1.hdf'
    evaluate(filename)