# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 14:10:50 2025

@author: user
"""
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

df_train = pd.read_csv("train.csv", parse_dates=['date']);

df_train.groupby("date")["sales"].sum().plot(figsize=(12, 4), title="Total Sales Over Time");


# =============================================================================
#time series for month  
# =============================================================================
df_train["month"] = df_train["date"].dt.month;

df_train["year"] = df_train["date"].dt.year

monthly_sales = df_train.groupby(["year", "month"])["sales"].mean().unstack("year")

monthly_sales.plot(figsize=(12, 5), title="Monthly Average Sales per Year")
plt.ylabel("Average Sales")
plt.grid(True)
plt.show()
