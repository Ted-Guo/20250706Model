# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 23:43:30 2025

@author: user
"""
import pandas as pd;
import matplotlib.pyplot as plt;
from scipy.stats import ttest_ind, f_oneway;
from statsmodels.tsa.seasonal import STL;

df = pd.read_csv("train.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday  # Monday=0 ... Sunday=6


#已將datetime 切割成date hour weekday
df = df.drop(columns = ["datetime"]);
dd = df.iloc[0];


#檢查缺失值
print(df.isna().sum());



# ===== EDA  檢查是否偏態=====
print(df[['temp','humidity','windspeed','count']].describe());





# =============================================================================
# # 假日 vs 非假日 箱型圖
# df.boxplot(column='count', by='holiday')
# plt.title("Holiday vs Non-Holiday Count")
# plt.suptitle("")
# plt.show()
# 
# # 工作日 vs 週末
# df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)
# df.boxplot(column='count', by='is_weekend')
# plt.title("Weekend vs Weekday Count")
# plt.suptitle("")
# plt.show()
# 
# # ===== t-test：假日 vs 非假日 =====
# holiday_cnt = df[df['holiday']==1]['count']
# normal_cnt  = df[df['holiday']==0]['count']
# t,p = ttest_ind(holiday_cnt, normal_cnt, equal_var=False)
# print(f"t-test 假日 vs 非假日 p值 = {p}")
# 
# # ===== ANOVA：weekday 影響 =====
# groups = [df[df['weekday']==i]['count'] for i in range(7)]
# F,p = f_oneway(*groups)
# print(f"ANOVA 星期幾之間 p值 = {p}")
# 
# # ===== STL 時間序列分解 =====
# ts = df.groupby('date')['count'].sum()   # 按天聚合
# ts.index = pd.to_datetime(ts.index)
# 
# stl = STL(ts, period=7)  # 假設weekly seasonality
# res = stl.fit()
# 
# fig = res.plot()
# plt.show()
# =============================================================================
