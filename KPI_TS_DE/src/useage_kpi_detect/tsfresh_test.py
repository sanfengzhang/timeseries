from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from win32api import GetDateFormat
download_robot_execution_failures()
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
import pandas as pd
import numpy as np
from pandas import Series
import  datetime


from tsfresh import extract_features
from tsfresh import extract_relevant_features


def getData():
    truth_df = pd.read_hdf('D:\\kpi\\1.hdf')
    kpi_names = truth_df['KPI ID'].values
    truth = truth_df[truth_df["KPI ID"] == kpi_names[0]]    
    y = truth['label']    
    data= truth.drop(columns=['KPI ID','label'])
    data=data.drop_duplicates(['timestamp'],keep='first') 
    data['timestamp']=data['timestamp'].map(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
   
    #ts = pd.DataFrame(data['value'].values, index=pd.DatetimeIndex(data['timestamp']),columns=['value'])
    
    return data,y

if __name__ == '__main__':
#==============================================================================
# # extract_features=extract_features(df2,column_id='id',column_sort='time')
# # print(extract_features.head())
# # 计算时间序列特征并提取最相关特征
# # f = open('F:/test.txt', 'w')  # 若是'wb'就表示写二进制文件
#==============================================================================
    data=getData()
    lendata=1000
    X=data[0][0:lendata]
    y=data[1][0:lendata]
    extract_settings = ComprehensiveFCParameters()
    extract_relevant_features = extract_features(X, column_id="timestamp", column_value="value", default_fc_parameters=extract_settings)
    arrs = extract_relevant_features.head()
    np.set_printoptions(threshold='nan')  # 全部输出
    pd.set_option('display.width', None)
    #print(extract_relevant_features)
    heads = extract_relevant_features.head()
    count=0
    for head in heads:       
        one_feature=extract_relevant_features[head][0:lendata]
        feature=one_feature.dropna()
        print(feature.size)
        if feature.size!=0:            
                suma=feature.sum()
                if suma!=0:
                    print(feature[0:10])
                    count=count+1
            

    print(count)
    
    #cwt_coefficients、fft_coefficient
    