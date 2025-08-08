# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 23:43:30 2025
全特徵 R2: 0.8334
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

print(df.describe());
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


# =============================================================================
# # 假設我們用 IQR 法找 count 的離群值
# Q1 = df['count'].quantile(0.25);
# Q3 = df['count'].quantile(0.75);
# IQR = Q3 - Q1;
# 
# 
# 
# 
# # ===== t-test：假日與非假日的平均租借量（count）是否有統計上顯著差異？ =====
# holiday_cnt = df[df['holiday']==1]['count'];
# normal_cnt  = df[df['holiday']==0]['count'];
# t,p = ttest_ind(holiday_cnt, normal_cnt, equal_var=False);
# print(f"t-test 假日與非假日的平均租借量（count）是否有統計上差異p值 = {p}");#p值 = 0.5461309933605478 ->沒有顯著差異
# 
# # =============================================================================
# # 
# # # 驗證: 看離群值中 holiday 的分布 is holiday 與count沒關 反而是  holiday = 0 比較多 (2:298)
# # outlier_mask = (df['count'] < (Q1 - 1.5 * IQR)) | (df['count'] > (Q3 + 1.5 * IQR));
# # outliers = df[outlier_mask];
# # print(outliers['holiday'].value_counts());
# # =============================================================================
# 
# 
# df = anova_funs.add_weekday_hour_feature(df);
# dd = df.iloc[0];
# 
# anova_funs.hitmap(df);
# anova_funs.hitmap_weekday_hour(df);
# 
# # ===== ANOVA：星期 影響 =====
# groups = [df[df['weekday']==i]['count'] for i in range(7)];
# F,p = f_oneway(*groups);
# print(f"ANOVA 星期幾之間 p值 = {p}");#p = 0.09219803980516852 > 0.05 但差異不大
# =============================================================================

#長條圖
#anova_funs.plot_week_chart(df);
#箱型圖
#anova_funs.plot_box(df, 'weekday', 'Weekday (0=Monday)', 'count', 'Count', 'Count Distribution by Weekday');

#get SS df MS
#Between Groups:60000, Within Groups:30000 波動太大
#anova_funs.get_anova_data(df);

#檢查hour
#anova_funs.plot_hour(df);

#檢查hour x weekday
#anova_funs.plot_hour_group_weekday(df);
#用anova table確認
#anova_funs.two_way_anova(df);



#交叉驗證 證明hitmap準確性 ANOVA / SS-MS, XGBoost importance
import statsmodels.api as sm;
from statsmodels.formula.api import ols;




# 單變項檢定
df = anova_funs.add_weekday_hour_feature(df);
# =============================================================================
# for col in ['weekday', 'hour', 'holiday', 'season', 'weekday_hour']:
#     anova_funs.run_anova(col, df)
# 
# =============================================================================


dd = df.iloc[0];

df = df.drop(columns=['date']);





from sklearn.model_selection import train_test_split;
from xgboost import XGBRegressor;
from sklearn.metrics import r2_score;
# 1️.OneHot encoding
categorical_cols = ['season', 'weekday', 'weekday_hour'];
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True);

# 把 registered 和 casual 移掉 因為coun = registered+casual 這樣會變成在看答案
X = df.drop(columns=['count', 'registered', 'casual']);
y = df['count'];

# 分訓練測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42);

# 建模型
model = XGBRegressor(random_state=42);
model.fit(X_train, y_train);

y_pred_full = model.predict(X_test);

score_full = r2_score(y_test, y_pred_full);
print(f"全特徵 R2: {score_full:.4f}");