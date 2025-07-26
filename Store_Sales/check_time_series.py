# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 20:45:49 2025

@author: user
"""

import pandas as pd;
import numpy as np;
from sklearn.cluster import KMeans;
from sklearn.preprocessing import StandardScaler;
from statsmodels.tsa.stattools import adfuller;

class time_series_function:
    #步驟 1：計算每個商品的銷售特徵
    def extract_features(self, df):
        agg = df.groupby('id')['sales'].agg(['mean', 'std', 'min', 'max']).fillna(0);
        return StandardScaler().fit_transform(agg);
    
    
    # 步驟 2：分群
    def cluster_sales(self, df, n_clusters=5):
        features = self.extract_features(df);
        km = KMeans(n_clusters=n_clusters, random_state=42);
        cluster_labels = km.fit_predict(features);
        return cluster_labels;

    # 步驟 3：每群取前30%代表（依據sales總和排序）
    def select_top_30_percent(self, df, cluster_labels):
        df = df.copy();
        df['cluster'] = cluster_labels[df['id'].values];
        
        selected_ids = [];
        for c in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == c];
            sales_sum = cluster_df.groupby('id')['sales'].sum();
            top_ids = sales_sum.sort_values(ascending=False);
            top_n = max(1, int(len(top_ids) * 0.3));
            selected_ids.extend(top_ids.head(top_n).index.tolist());
        
        return df[df['id'].isin(selected_ids)];
    
    # 步驟 4：將所有銷售序列依時間拼接（模擬整體趨勢）
    def concat_sales(self, df):
        df_sorted = df.sort_values(by=['date']);
        series = df_sorted.groupby('date')['sales'].sum();
        return series;


    # 步驟 5：做 ADF test
    def run_adf(self, series):
        result = adfuller(series.dropna());
        print(f"ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}");
        print("Critical Values:");
        for key, value in result[4].items():
            print(f"    {key}: {value:.4f}");
        return result;
        


 
