import pandas as pd

from pandas import Series, DataFrame

index = pd.date_range('2017-08-15', periods=10)
data = Series(list(range(10)), index=index)

df=data.to_frame("my_df").reindex()

print(df.dropna())



index1 = pd.date_range('2017-08-15', periods=100)
data1 = Series(list(range(100)), index=index1)

for indexs in data.index:
    key = str(indexs)
    value = str(data[indexs])
  #  print(key+" "+value)     
