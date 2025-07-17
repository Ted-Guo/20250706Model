# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 10:09:05 2025

@author: user
"""

import pandas as pd;
import numpy as np;


# =============================================================================
# family:商品種類 date:日期 sales:交易金額(0代表沒賣出去) onpromotion:是否促銷 store_nbr:店號
# =============================================================================
df_train = pd.read_csv("train.csv", parse_dates=['date']);


# =============================================================================
# store_nbr:店號 city:城市 state:州  type:分店類型（A~D） cluster:集群編號（0~16）用某種相似性依據（例如地區、顧客行為、商品組合、營收等）分成數個群組
# =============================================================================
df_stores = pd.read_csv("stores.csv");


# =============================================================================
# date:日期 store_nbr:店號 transactions:當日交易數
# =============================================================================
df_transactions = pd.read_csv("transactions.csv", parse_dates=['date']);


# =============================================================================
# date:日期 type:假期類型 locale:假期影響範圍 locale_name:地點名稱 decription:假日名稱 transferred: true->「雖然本來是節日，但實際不是放假日」
# =============================================================================
df_holidays = pd.read_csv("holidays_events.csv", parse_dates=['date']);


# =============================================================================
# date:日期 dcoilwtico:油價
# =============================================================================
df_oil = pd.read_csv("oil.csv", parse_dates=['date']);

# =============================================================================
# check data
# =============================================================================
df_head = df_train.head();
df_train['date'].min();
df_train['date'].max();

# =============================================================================
# merge df
# =============================================================================
print("==============train=====================");
print("train",df_train.iloc[0]);
print("==============store=====================");
print("store",df_stores.iloc[0]);

df_train = df_train.merge(df_stores,how="left",on = "store_nbr");
print("==============merge store=====================");
print("merge store",df_train.iloc[0]);
df_train = df_train.merge(df_oil,how = "left",on = "date");
print("==============merge oil=====================");
print("merge oil",df_train.iloc[0]);

df_holidays = df_holidays[df_holidays['transferred'] == False];
# =============================================================================
# 留下要用欄位
# =============================================================================
df_holidays = df_holidays[['date', 'type', 'locale', 'locale_name', 'description']];

# =============================================================================
# 同一天可能有兩個以上假期 將他們合併 ex:2017-12-22 00:00:00
# =============================================================================
df_holidays = df_holidays.groupby('date').agg({
    'type': lambda x: ','.join(set(x)),
    'locale': lambda x: ','.join(set(x)),
    'description': lambda x: ','.join(set(x))
}).reset_index();

df_train = df_train.merge(df_holidays,how = "left",on = "date");
print("==============merge holidays=====================");
print("merge holidays",df_train.iloc[0]);
