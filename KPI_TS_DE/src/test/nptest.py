# coding=utf-8
import pandas as pd
import numpy as np

df = pd.DataFrame({'a':np.random.randint(1, 100, 10)})

print(df)
                  
def ts_diff(ts, n):
    count = 0  
    while(count < n):
        ts = ts.diff(1)
        ts.dropna(inplace=True)
        count = count + 1        
    return ts



#===============================================================================
# print(df.diff().diff())
# print(ts_diff(df, 2))
#===============================================================================


lendf = int(len(df))
last = df.iloc[lendf - 1:lendf]

#print(last,type(last))



a = pd.Series([1,3,5],index = ['a','b','c'])


print(a)
print(a['a'],a['b'],a.keys()[0])
print(a[1:3])
b=a[1:3]
print(b.sum())

a['d']=7

print(a)
print(3.005711e-05)


