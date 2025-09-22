# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 12:16:19 2025

@author: user
"""

import pandas as pd;
import numpy as np;


#檢查train 資料型態 有無缺失值
df_train = pd.read_csv("train.csv", parse_dates=['date']);
df = df_train.iloc[0];

# =============================================================================
#  0   id           int64         
#  1   date         datetime64[ns]
#  2   store_nbr    int64         
#  3   family       object        
#  4   sales        float64       
#  5   onpromotion  int64  
# =============================================================================
print("==================train.csv==================================");
print(df_train.dtypes);
print("==============================================================");
# 每個欄位缺失值數量
print(df_train.isnull().sum());
print("==============================================================");



#檢查transactions 資料型態 有無缺失值
df_transactions = pd.read_csv("transactions.csv", parse_dates=['date']);

print("==================transactions.csv==================================");
# =============================================================================
# date            datetime64[ns]
# store_nbr                int64
# transactions             int64
# =============================================================================
print(df_transactions.dtypes);
print("==============================================================");
# 每個欄位缺失值數量
print(df_transactions.isnull().sum());
print("=======================end=============================");


#檢查holidays_events 資料型態 有無缺失值
df_holidays = pd.read_csv("holidays_events.csv", parse_dates=['date']);

print("==================holidays_events.csv==================================");
# =============================================================================
# date           datetime64[ns]
# type                   object
# locale                 object
# locale_name            object
# description            object
# transferred              bool
# =============================================================================
print(df_holidays.dtypes);
print("==============================================================");
# 每個欄位缺失值數量
print(df_holidays.isnull().sum());
print("=======================end=============================");


#oil 資料型態 有無缺失值
df_oil = pd.read_csv("oil.csv", parse_dates=['date']);

print("==================oil.csv==================================");
# =============================================================================
# date          datetime64[ns]
# dcoilwtico           float64
# =============================================================================
print(df_oil.dtypes);
print("==============================================================");
# 每個欄位缺失值數量
print(df_oil.isnull().sum());#dcoilwtico:43
print("==============================================================");

# =============================================================================
# 處理缺失值，油價是線性的，前後一天差異不大，使用:
# 1.向前填值法，如果沒有前一筆資料則使用向後填值法
# 2.interpolate
# =============================================================================
df_oil['dcoilwtico'] = df_oil['dcoilwtico'].interpolate(method='linear');
print("===================after interpolate=================================");
print(df_oil.isnull().sum());#1筆缺失值
print(df_oil[df_oil['dcoilwtico'].isna()]);#資料欄位第一筆
#使用 backward fill
df_oil['dcoilwtico'] = df_oil['dcoilwtico'].bfill();
print("===================after backward fill=================================");
print(df_oil[df_oil['dcoilwtico'].isna()]);
print("=======================end=============================");


#檢查stores 資料型態 有無缺失值
df_stores = pd.read_csv("stores.csv");

print("==================stores.csv==================================");
# =============================================================================
# store_nbr     int64
# city         object
# state        object
# type         object
# cluster       int64
# =============================================================================
print(df_stores.dtypes);
print("==============================================================");
# 每個欄位缺失值數量
print(df_stores.isnull().sum());
print("=======================end=============================");
























