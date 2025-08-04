# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 23:43:30 2025

@author: user
"""
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
from scipy.stats import ttest_ind, f_oneway;
from statsmodels.tsa.seasonal import STL;

df = pd.read_csv("train.csv");
df['datetime'] = pd.to_datetime(df['datetime']);
df['date'] = df['datetime'].dt.date;
df['hour'] = df['datetime'].dt.hour;
df['weekday'] = df['datetime'].dt.weekday;  # Monday=0 ... Sunday=6


#已將datetime 切割成date hour weekday
df = df.drop(columns = ["datetime"]);
dd = df.head();


#檢查缺失值
#print(df.isna().sum());



# ===== EDA  檢查是否偏態=====
#print(df[['temp','humidity','windspeed','count']].describe());


plt.figure(figsize=(12,5))
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'];
plt.rcParams['axes.unicode_minus'] = False;


# 直方圖
plt.subplot(1,2,1);
sns.histplot(df['count'], bins=60, kde=True);  # kde=True 會畫密度線
plt.xlabel('count');
plt.ylabel('資料筆數');
plt.title('Histogram and KDE of count');

# 箱型圖 
plt.subplot(1,2,2);
sns.boxplot(y=df['count']);
plt.ylabel('count');
plt.title('Boxplot of count');

plt.tight_layout();
plt.show();


# 假設我們用 IQR 法找 count 的離群值
Q1 = df['count'].quantile(0.25);
Q3 = df['count'].quantile(0.75);
IQR = Q3 - Q1;

# 判斷離群值跟假日有沒有關係
outlier_mask = (df['count'] < (Q1 - 1.5 * IQR)) | (df['count'] > (Q3 + 1.5 * IQR));
outliers = df[outlier_mask];

# 看離群值中 holiday 的分布
print(outliers['holiday'].value_counts());





