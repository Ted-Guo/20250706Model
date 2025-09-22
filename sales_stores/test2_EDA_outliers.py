# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 13:19:05 2025

@author: user
"""

import pandas as pd;
import numpy as np;

import seaborn as sns;
import matplotlib.pyplot as plt;

df_train = pd.read_csv("train.csv", parse_dates=['date']);
# =============================================================================
# print(df_train.dtypes);
# print(df_train["onpromotion"].describe());#當天促銷商品數量
# #大多沒有促銷 只有少部分幾天有，且一次數量龐大
# sns.histplot(df_train['onpromotion'], bins=50, log_scale=(False, True));
# plt.show();
# print(df_train["onpromotion"].describe());
# =============================================================================



#簡易分組，是否有促銷flag
df_train["Promo_flag"] = df_train['onpromotion'].apply(lambda x: 0 if x == 0 else 1);


# 1. 計算店家 lift
df_store_promo = df_train.groupby(['store_nbr','Promo_flag'])['sales'].mean().unstack();
df_store_promo['lift'] = (df_store_promo[1]-df_store_promo[0]) / df_store_promo[0] * 100;

# 2. 取前10名 lift
top10 = df_store_promo.sort_values('lift', ascending=False).head(10).reset_index();

# 3. 讀取店家資料
df_stores = pd.read_csv('stores.csv');

# 4. merge 店名資訊
top10 = top10.merge(df_stores, on='store_nbr', how='left');

# 5. 看結果
#調整欄位名
top10['lift'] = top10['lift'].round(2);
top10.rename(columns={0:'No Promotion', 1:'Promotion'}, inplace=True);
cols = ['No Promotion','Promotion','lift','city','state'];
print(top10[cols]);

#畫圖
plt.figure(figsize=(12,6));

# 長條圖
sns.barplot(data=top10, x='city', y='lift', palette='viridis');


# 先建立顯示用的欄位：city + store_nbr
top10['store_label'] = top10['city'] + ' (' + top10['store_nbr'].astype(str) + ')';

plt.figure(figsize=(14,7));

# 長條圖，bar 顏色依 lift 高低
sns.barplot(data=top10, x='store_label', y='lift', palette='viridis');

# 標籤與標題
plt.ylabel('Lift (%)');
plt.xlabel('Store (City)');
plt.title('Top 10 Stores by Promotion Lift');
plt.xticks(rotation=45, ha='right');

# 在每個 bar 上加上非促銷 -> 促銷平均銷售額
for index, row in top10.iterrows():
    plt.text(index, row['lift'] + 200, f"{row['No Promotion']:.1f} → {row['Promotion']:.0f}", 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout();
plt.show();


# =============================================================================
# 分析結論（基於 Top10 lift 圖表）
# =============================================================================
# 1️. 促銷效果明顯的店家
# - Lift 高的店表示促銷對銷售額提升效果非常大
# - 例如 stor_nbr 52：非促銷日平均不到 4 元，促銷日衝到 1271 元 → 爆炸性提升
# - 建議：低基數店家促銷可快速拉升，但需注意波動性

# 2️. 絕對銷售額 vs 百分比提升
# - 除了 lift 百分比，也要看非促銷與促銷的絕對銷售額
# - 有些店 lift 高，但促銷前銷售額極低 → 絕對增幅有限
# - 有些店 lift 中等，但促銷後銷售額高 → 絕對收益可觀
# - 建議同時考慮 lift + 絕對增幅

# 3️. 城市或區域差異
# - 可看圖表 x 軸城市標籤，判斷區域促銷效果
# - 例如 Manta、Guayas 多次出現 → 這些區域促銷可能更有效

# 4️. 店家類型或群組分析
# - 若有 type / cluster 欄位，可分析大型連鎖 vs 小型店、不同 cluster 促銷效果
# - 幫助制定差異化策略

# 5️. 發現數據異常或極端值
# - 超高 lift (>1000%) 通常是非促銷平均過小
# - 可篩選極端數據，避免誤判

# 6️. 策略建議
# - 低基數高 lift 店：短期促銷快速拉升，但注意波動
# - 高基數中 lift 店：促銷帶來穩定收益
# - 可優先選擇 lift 高且促銷後絕對銷售額高的店做重點投放

#  總結
# - 圖表回答三個核心問題：
#   1. 哪些店促銷效果最好（百分比 + 絕對值）
#   2. 哪些城市 / 區域反應最好
#   3. 哪些店可能有數據異常或需注意

