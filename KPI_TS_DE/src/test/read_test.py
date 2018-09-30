import pandas as pd

import time as time
import datetime as datetime
import json

def getData():
    truth_df = pd.read_hdf('D:\\kpi\\1.hdf')
    kpi_names = truth_df['KPI ID'].values
    truth = truth_df[truth_df["KPI ID"] == kpi_names[0]]  
    
    jsonData = {'timestamp':'' , 'value': "", 'label': ''}
    file = open("D:\\kpi\\result_tmp.txt", 'w')
    for  row in truth.iterrows():
        jsonData['timestamp'] = row[1]['timestamp']
        jsonData['value'] = row[1]['value']   
        jsonData['label'] = row[1]['label']       
        file.write(json.dumps(jsonData, ensure_ascii=False) + '\n')
    file.close()
    
    data= truth.drop(columns=['label', 'KPI ID'])
    data['timestamp']=data['timestamp'].map(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))


    
    ts = pd.Series(data['value'].values, index=data['timestamp'])
    print(ts)
  
    return ts



getData()

now = 1538117085

date=datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S.%S')
print(date)

a=1482941460
b=1482941520

aa=datetime.datetime.fromtimestamp(a).strftime('%Y-%m-%d %H:%M:%S.%S')
print(aa)
bb=datetime.datetime.fromtimestamp(b).strftime('%Y-%m-%d %H:%M:%S.%S')
print(bb)

