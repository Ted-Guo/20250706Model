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
from anova import anova_fun;


anova_funs = anova_fun();
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



# =============================================================================
# # ===== EDA  檢查是否偏態=====
# #print(df[['temp','humidity','windspeed','count']].describe());
# 
# 
# plt.figure(figsize=(12,5))
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'];
# plt.rcParams['axes.unicode_minus'] = False;
# 
# 
# # 直方圖
# plt.subplot(1,2,1);
# sns.histplot(df['count'], bins=60, kde=True);  # kde=True 會畫密度線
# plt.xlabel('count');
# plt.ylabel('資料筆數');
# plt.title('Histogram and KDE of count');
# 
# # 箱型圖 
# plt.subplot(1,2,2);
# sns.boxplot(y=df['count']);
# plt.ylabel('count');
# plt.title('Boxplot of count');
# 
# plt.tight_layout();
# plt.show();
# =============================================================================


# 假設我們用 IQR 法找 count 的離群值
Q1 = df['count'].quantile(0.25);
Q3 = df['count'].quantile(0.75);
IQR = Q3 - Q1;




# ===== t-test：假日與非假日的平均租借量（count）是否有統計上顯著差異？ =====
holiday_cnt = df[df['holiday']==1]['count'];
normal_cnt  = df[df['holiday']==0]['count'];
t,p = ttest_ind(holiday_cnt, normal_cnt, equal_var=False);
print(f"t-test 假日與非假日的平均租借量（count）是否有統計上差異p值 = {p}");#p值 = 0.5461309933605478 ->沒有顯著差異

# =============================================================================
# 
# # 驗證: 看離群值中 holiday 的分布 is holiday 與count沒關 反而是  holiday = 0 比較多 (2:298)
# outlier_mask = (df['count'] < (Q1 - 1.5 * IQR)) | (df['count'] > (Q3 + 1.5 * IQR));
# outliers = df[outlier_mask];
# print(outliers['holiday'].value_counts());
# =============================================================================



# ===== ANOVA：星期 影響 =====
groups = [df[df['weekday']==i]['count'] for i in range(7)];
F,p = f_oneway(*groups);
print(f"ANOVA 星期幾之間 p值 = {p}");#p = 0.09219803980516852 > 0.05 但差異不大


#get SS df MS
anova_funs.get_anova_data(df);

#長條圖
anova_funs.plot_week_chart(df);


#箱型圖
anova_funs.plot_box(df, 'weekday', 'Weekday (0=Monday)', 'count', 'Count', 'Count Distribution by Weekday');


weekday_std = df.groupby('weekday')['count'].std().reset_index()

# 畫圖
# 同時畫出 mean 和 std（誤差棒）
weekday_stats = df.groupby('weekday')['count'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=weekday_stats, x='weekday', y='mean', color='skyblue', label='Mean')
plt.errorbar(x=weekday_stats['weekday'],
             y=weekday_stats['mean'],
             yerr=weekday_stats['std'],
             fmt='none', capsize=5, color='red', label='± Std')

plt.title("Weekday Mean Count with Standard Deviation")
plt.xlabel("Weekday (0=Mon, 6=Sun)")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.show()
