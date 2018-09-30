# coding=utf-8
import pandas as pd
import numpy as np

df = pd.DataFrame({'a':np.random.randint(1, 100, 10)})

print(df)

df=df.sort_values(by="a" , ascending=True)

print(df)

print('-----------')
                  
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


s = pd.Series([1,2,3,4,5],index=["a","a","g","b","c"])
print(s)
print(s.index.is_unique)

s=s.drop_duplicates(keep='first') 
print(s)
print(s.index.is_unique)


df1 = pd.DataFrame({'a':{1,2,4},'b':np.random.randint(1, 100, 3)})

print(df1)

s1 = pd.Series([1,2,3,4,5],index=["a","a","g","b","c"],name='aa')

print(s1.sum())





