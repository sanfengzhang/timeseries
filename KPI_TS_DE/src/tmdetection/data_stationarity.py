# coding=utf-8
from statsmodels.tsa.stattools import adfuller
    
    
def  stationarity_result_analyse(tsdata):
    resultTuple = dostationarity(tsdata)    
    print(resultTuple)
    adfTestResult = resultTuple[0]
    criticalValuelResult = resultTuple[4]
    criticalValuelResult1 = criticalValuelResult['1%']
    p_value = resultTuple[1]
    b = 1 / 100000000000
    if adfTestResult < criticalValuelResult1 and  p_value < b:
        return True
    else:
        return False
    

def dostationarity(tsdata):    
    resultADF = adfuller(tsdata.dropna(), 1)
    return resultADF

