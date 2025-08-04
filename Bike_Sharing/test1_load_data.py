# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 23:43:30 2025

@author: user
"""
import pandas as pd;
import matplotlib.pyplot as plt;
from scipy.stats import ttest_ind, f_oneway;
from statsmodels.tsa.seasonal import STL;

df = pd.read_csv("train.csv");
df['datetime'] = pd.to_datetime(df['datetime']);
df['date'] = df['datetime'].dt.date;
df['hour'] = df['datetime'].dt.hour;
df['weekday'] = df['datetime'].dt.weekday;  # Monday=0 ... Sunday=6


#已將datetime 切割成date hour weekday
df = df.drop(columns = ["datetime"]);
dd = df.iloc[0];


#檢查缺失值
print(df.isna().sum());



# ===== EDA  檢查是否偏態=====
print(df[['temp','humidity','windspeed','count']].describe());





