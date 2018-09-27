from sklearn import preprocessing 
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import pandas as pd
import json

from imblearn.combine import SMOTEENN

 
#===============================================================================
# 归一化----将数据特征缩放至某一范围(scalingfeatures to a range)
# 
# 　　另外一种标准化方法是将数据缩放至给定的最小值与最大值之间，通常是０与１之间，可用MinMaxScaler实现。或者将最大的绝对值缩放至单位大小，可用MaxAbsScaler实现。
# 
# 使用这种标准化方法的原因是，有时数据集的标准差非常非常小，有时数据中有很多很多零（稀疏数据）需要保存住０元素。
# 
#  　　2.1 MinMaxScaler(最小最大值标准化)
# 
# 　　公式：X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) ; 
# 
# 　　　　　X_scaler = X_std/ (max - min) + min
#===============================================================================
def minMaxScale(): 
# 例子：将数据缩放至[0, 1]间
    X_train = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])
    min_max_scaler = preprocessing.MinMaxScaler() 
    X_train_minmax = min_max_scaler.fit_transform(X_train) 
    print(X_train_minmax)
    X_test = np.array([[ -3., -1., 4.]])  
    X_test_minmax = min_max_scaler.transform(X_test)
    print(X_test_minmax)
    print(min_max_scaler.scale_)     
    print(min_max_scaler.min_)  
    

#===============================================================================
# 在实际应用中，读者可能会碰到一种比较头疼的问题，那就是分类问题中类别型的因变量可能存在严重的偏倚，即类别之间的比例严重失调。
# 如欺诈问题中，欺诈类观测在样本集中毕竟占少数；客户流失问题中，非忠实的客户往往也是占很少一部分；
# 在某营销活动的响应问题中，真正参与活动的客户也同样只是少部分。
# 
# 如果数据存在严重的不平衡，预测得出的结论往往也是有偏的，即分类结果会偏向于较多观测的类。
# 对于这种问题该如何处理呢？最简单粗暴的办法就是构造1:1的数据，要么将多的那一类砍掉一部分（即欠采样），
# 要么将少的那一类进行Bootstrap抽样（即过采样）。但这样做会存在问题，对于第一种方法，
# 砍掉的数据会导致某些隐含信息的丢失；而第二种方法中，有放回的抽样形成的简单复制，又会使模型产生过拟合。
#===============================================================================
#===============================================================================
# SMOTE算法的基本思想就是对少数类别样本进行分析和模拟，并将人工模拟的新样本添加到数据集中，
# 进而使原始数据中的类别不再严重失衡。该算法的模拟过程采用了KNN技术，模拟生成新样本的步骤如下： 
# 采样最邻近算法，计算出每个少数类样本的K个近邻；
# 从K个近邻中随机挑选N个样本进行随机线性插值；
# 构造新的少数类样本；
# 将新样本与原数据合成，产生新的训练集；
#===============================================================================
def smpote_test():    
    # 读取测试测试数据集中的数据
    truth_df = pd.read_hdf('D:\\kpi\\1.hdf')
    # print(truth_df["KPI ID"])
    kpi_names = truth_df['KPI ID'].values
    truth = truth_df[truth_df["KPI ID"] == kpi_names[0]]
    y = truth['label']
    
    X = truth.drop(columns=['label', 'KPI ID'])   
    sm = SMOTEENN()
    X_resampled, y_resampled = sm.fit_sample(X, y)   
   
    
    dfX = pd.DataFrame(X_resampled,columns=['timestamp','value'])
    DFy=pd.DataFrame(y_resampled,columns=['label'])
   
    plt.plot(np.array(X['timestamp']), np.array(X['value']), color='green', label='training accuracy')
    plt.legend() # 显示图例
    plt.show() 
      
    dfX=dfX.join(DFy).sort_values(by="timestamp" , ascending=True)
    jsonData = {'timestamp': '', 'value': "", 'label': ''}
    file = open("D:\\kpi\\result_tmp.txt", 'w')
    for  row in dfX.iterrows():
        jsonData['timestamp'] = row['timestamp']
        jsonData['value'] = row['value']   
        jsonData['label'] = row['label']       
        file.write(json.dumps(jsonData, ensure_ascii=False) + '\n')

    file.close()
    print("end....")




              
smpote_test()   

#===============================================================================
# X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
#                            n_informative=3, n_redundant=1, flip_y=0,
#                            n_features=20, n_clusters_per_class=1,
#                            n_samples=100, random_state=10)
# print(X)
# print(y)
# print(y.shape)
# sm = SMOTEENN()
# X_resampled, y_resampled = sm.fit_sample(X, y)
# print(y_resampled)
# print(y_resampled.shape)
#===============================================================================
