# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 20:26:03 2025

@author: user
"""

import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;


df_train = pd.read_csv("train.csv", parse_dates=['date']);

#生成Promo_flag
df_train["Promo_flag"] = df_train['onpromotion'].apply(lambda x: 0 if x == 0 else 1);


#統計 促銷時sales 平均值
store_promo_sales = (
    df_train[df_train['Promo_flag']==1]
    .groupby('store_nbr')['sales']
    .mean()
    .reset_index()
    .rename(columns={'sales':'Promotion'})
);

#統計 非促銷時sales 平均值
store_no_promo_sales = (
    df_train[df_train['Promo_flag']==0]
    .groupby('store_nbr')['sales']
    .mean()
    .reset_index()
    .rename(columns={'sales':'No Promotion'})
);

#算lift
top10_revenue = store_promo_sales.merge(store_no_promo_sales, on='store_nbr');
top10_revenue['lift'] = (top10_revenue['Promotion'] - top10_revenue['No Promotion']) / top10_revenue['No Promotion'] * 100;



# merge 城市資訊
df_stores = pd.read_csv("stores.csv").drop_duplicates(subset='store_nbr')
top10_revenue = top10_revenue.merge(df_stores[['store_nbr','city']], on='store_nbr', how='left');
top10_revenue = top10_revenue.sort_values('Promotion', ascending=False).head(10);
print(top10_revenue);
# 建立 store_label
top10_revenue['store_label'] = top10_revenue['store_nbr'].astype(str) + " (" + top10_revenue['city'] + ")";



plt.figure(figsize=(12,6));
ax = sns.barplot(data=top10_revenue, x='store_label', y='Promotion', palette='viridis');
plt.ylabel('Average Sales during Promotion');
plt.xlabel('Store (City)');
plt.title('Top 10 Stores by Absolute Sales during Promotion');

# 標註文字
for bar, row in zip(ax.patches, top10_revenue.itertuples()):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 50, 
            f"{row._3:.1f} → {row.Promotion:.0f}",  # row._3 是 No Promotion，根據 itertuples 位置
            ha='center', va='bottom', fontsize=10)

plt.ylabel('Average Sales during Promotion');
plt.xlabel('Store (City)');
plt.title('Top 10 Stores by Absolute Sales during Promotion');
plt.xticks(rotation=45, ha='right');
plt.tight_layout();
plt.show();



# =============================================================================
# 分析結論（基於 Top10 revenue 圖表）
# =============================================================================
# 1️. 了解最賺錢的是哪些店家
# 2.前10名之中有8個坐落在相同城市->該城市消費力驚人
# 3.雖然這些不是lift成長力最高的，但是卻是基數最高的店家
# 4.top10 lift + top 10 revenue 可以得到
#   a.高絕對收益的穩定賺錢店
#   b.高成長率的小店（短期爆發效果好）
# =============================================================================

