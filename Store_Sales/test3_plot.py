# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 14:10:50 2025

@author: user
"""
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import gzip
import pickle


from data_preprocessor import data_functions;
from check_time_series import time_series_function;
from soft_clustering import trans_locale_function;



common_functions = data_functions();
time_functions = time_series_function();
locale_function =  trans_locale_function();
df_train = common_functions.get_df_after_handled();

#雖然整年間有上上下下，但是每年都有往上攀升趨勢 -> sales 存在趨勢性(Trend)與季節性(Seasonality)
#df_train.groupby("date")["sales"].sum().plot();


# =============================================================================
# #周末銷售特別高
# group = df_train.groupby("dayofweek")["sales"].mean();
# group.plot(kind = "bar",title = "Average Sales by Day of Week");
# plt.xlabel("Day of Week");
# plt.ylabel("Average Sales");
# plt.show();
# 
# #12月銷售特別高
# group_mon = df_train.groupby("month")["sales"].mean();
# group_mon.plot(kind = "bar",title = "Average Sales by Day of Week");
# plt.xlabel("Month");
# plt.ylabel("Average Sales");
# plt.show();
# =============================================================================

# =============================================================================
# #觀察是否周末銷售高是否有週期性
# group = df_train[df_train["year"] == 2013 ].groupby(["dayofweek"])["sales"].mean();
# group.plot(kind = "bar",title = "2013 year everage Sales by Day of Week");
# plt.xlabel("Day of Week");
# plt.ylabel("Average Sales");
# plt.show();
# 
# group = df_train[df_train["year"] == 2014 ].groupby(["dayofweek"])["sales"].mean();
# group.plot(kind = "bar",title = "2014 year everage Sales by Day of Week");
# plt.xlabel("Day of Week");
# plt.ylabel("Average Sales");
# plt.show();
# 
# group = df_train[df_train["year"] == 2015 ].groupby(["dayofweek"])["sales"].mean();
# group.plot(kind = "bar",title = "2015 year everage Sales by Day of Week");
# plt.xlabel("Day of Week");
# plt.ylabel("Average Sales");
# plt.show();
# 
# group = df_train[df_train["year"] == 2016 ].groupby(["dayofweek"])["sales"].mean();
# group.plot(kind = "bar",title = "2016 year everage Sales by Day of Week");
# plt.xlabel("Day of Week");
# plt.ylabel("Average Sales");
# plt.show();
# 
# 
# group = df_train[df_train["year"] == 2017 ].groupby(["dayofweek"])["sales"].mean();
# group.plot(kind = "bar",title = "2017 year everage Sales by Day of Week");
# plt.xlabel("Day of Week");
# plt.ylabel("Average Sales");
# plt.show();
# =============================================================================


# =============================================================================
# #week轉換成折線圖
# plt.figure(figsize=(10, 6));
# 
# # 針對每一年，畫出每個 dayofweek 的平均 sales 折線圖
# for year in range(2013, 2018):
#     group = df_train[df_train["year"] == year].groupby("dayofweek")["sales"].mean();
#     plt.plot(group.index, group.values, marker="o", label=str(year));
# 
# # 加上標籤與標題
# plt.title("Average Sales by Day of Week (2013-2017)");
# plt.xlabel("Day of Week (0=Sunday, 6=Saturday)");
# plt.ylabel("Average Sales");
# plt.legend(title="Year");
# plt.grid(True);
# plt.xticks(ticks=[0,1,2,3,4,5,6], labels=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]);
# plt.tight_layout();
# plt.show();
# =============================================================================



# =============================================================================
# #month轉換成折線圖
# plt.figure(figsize=(10, 6));
# 
# # 針對每一年，畫出每個 dayofweek 的平均 sales 折線圖
# for year in range(2013, 2018):
#     group = df_train[df_train["year"] == year].groupby("month")["sales"].mean();
#     plt.plot(group.index, group.values, marker="o", label=str(year));
# 
# # 加上標籤與標題
# mon_ticks = [1,2,3,4,5,6,7,8,9,10,11,12];
# mon_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
#           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
# plt.title("Average Sales by month (2013-2017)");
# plt.xlabel("month");
# plt.ylabel("Average Sales");
# plt.legend(title = "Year");
# plt.grid(True);
# plt.xticks(ticks = mon_ticks, labels = mon_labels);
# plt.tight_layout();
# plt.show();
# 
# =============================================================================


#確認預估目標是否平穩來決定模型類型
# =============================================================================
# df_sales = df_train.sort_values("date")
# cluster_labels = time_functions.cluster_sales(df_sales, n_clusters=5)
# df_selected = time_functions.select_top_30_percent(df_sales, cluster_labels)
# overall_series = time_functions.concat_sales(df_selected)
# time_functions.run_adf(overall_series);
# 
# =============================================================================



# =============================================================================
# print(list(df_train.columns));
# print(df_train.iloc[0]);
# =============================================================================



# =============================================================================
# desc_counts = df_train['description'].value_counts()
# print(desc_counts)
# print(f"總共有 {len(desc_counts)} 種不同的節日描述")
# =============================================================================


df_train = common_functions.create_multi_label_holiday_features(df_train);
#print(df_train.iloc[0]);
# =============================================================================
# df_train = common_functions.add_lag_rolling_features(
#     df=df_train,
#     group_keys=['store_nbr', 'family'],
#     target_cols=['sales', 'transactions'],
#     lags=[1, 7, 14],
#     rolling_windows=[7, 14]
# );
# =============================================================================



#save data
locale_function.description_sentence_trans(df_train);

df = df_train.iloc[0];