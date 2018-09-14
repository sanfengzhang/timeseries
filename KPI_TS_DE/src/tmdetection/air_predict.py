from tmdetection.air_ARIMA_model import get_ARMA as air_ARMA
import pandas as pd
import numpy as np


def data():    
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
   
    data = pd.read_csv('C:\\Users\\owner\\Desktop\\AirPassengers.csv', parse_dates=['Month'], index_col='Month',
                   date_parser=dateparse)
    data.index = pd.DatetimeIndex(data.index.values,
                               freq=data.index.inferred_freq)
    ts = data['#Passengers']
    return ts


def predict(start, end):
    # 这个按模型计算出来的是一阶差分序列对应的一阶差分值
   
    predict_ts_2_news = air_arma_mode.predict(start, end)    
    global ts_data_roll_mean_diff 
    global ts_data_roll_mean
   
    pre_test_data = {}
    len_predict_ts_2_news = int(len(predict_ts_2_news))
    for i in range (len_predict_ts_2_news):
        predict_ts_2_new = predict_ts_2_news[i:i + 1]
        key = predict_ts_2_new.keys()[0]
        pre_value = predict_ts_2_new[key]
        
        last_ts_data_roll_mean_diff = ts_data_roll_mean_diff[-1]        
        pre_ts_data_roll_mean_diff = pre_value + last_ts_data_roll_mean_diff
        ts_data_roll_mean_diff[key] = pre_ts_data_roll_mean_diff
        
        # 计算移平均值的差分项数据、将取到的MA的最后一条数据和预测出来的MA对应一阶差分值进行求和计算。结果就是预测这个月对应的MA
        # 将计算出来的结果更新到MA中去
        last_ts_data_roll_mean = ts_data_roll_mean[-1]
        new_ts_data_roll_mean = last_ts_data_roll_mean + pre_ts_data_roll_mean_diff
        ts_data_roll_mean[key] = new_ts_data_roll_mean       
        
        # 根据MA计算方法计算预测月的值
        len_ts_data_log = int(len(ts_data_log))
        suma = ts_data_log[len_ts_data_log - 11:len_ts_data_log].sum()
        pre_ts_data_roll_mean = 12 * new_ts_data_roll_mean - suma
        ts_data_log[key] = pre_ts_data_roll_mean        
        pre_test_data[key] = int(np.exp(pre_ts_data_roll_mean))
   
    #计算预测值和实际值的RMES均方差平方根误差值   ARMA RMSE: 5.1235
    pre_test_data_ser = pd.Series(pre_test_data, name='pre_test_data')  
    print('ARMA RMSE: %.4f' % np.sqrt(sum((pre_test_data_ser - test_data) ** 2) / test_data.size))   

    
data = data()
train_size = int(len(data) * 0.67)   
ts_data = data[0:train_size]
test_data = data[train_size:train_size + 12]
result = air_ARMA(ts_data, 12) 
air_arma_mode = result[0]
ts_data_roll_mean = result[2]
ts_data_roll_mean_diff = result[1]
ts_data_log = result[3]
    
predict('1957-01-01', '1957-12-01')   



