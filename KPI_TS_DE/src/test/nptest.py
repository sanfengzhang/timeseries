# coding=utf-8
import pandas as pd
import numpy as np

df = pd.DataFrame({'a':np.random.randint(1, 100, 10)})
                  
def ts_diff(ts, n):
    count=0  
    while(count<n):
        ts=ts.diff(1)
        ts.dropna(inplace=True)
        count = count + 1        
    return ts


print(df.diff().diff())

print(ts_diff(df, 2))

