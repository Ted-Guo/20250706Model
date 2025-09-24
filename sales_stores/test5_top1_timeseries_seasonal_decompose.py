import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
import os;

from statsmodels.tsa.seasonal import seasonal_decompose;
from matplotlib.dates import DateFormatter;

# 1. 讀取資料
df_train = pd.read_csv("train.csv", parse_dates=['date']);
df_revenue = pd.read_parquet("top10_revenue.parquet");  # Top10 Revenue 店家

# 生成促銷 flag
df_train['Promo_flag'] = df_train['onpromotion'].apply(lambda x: 0 if x == 0 else 1);

# 篩選 Top10 店家
df_top10_daily = df_train[df_train['store_nbr'].isin(df_revenue['store_nbr'])].copy();


#df_revenue 已經排序過 先觀察top1的時間趨勢
top1_id = int(df_revenue.iloc[0]['store_nbr']);
df_top1 = df_train[df_train['store_nbr'] == top1_id].copy();

df_top1['year']  = df_top1['date'].dt.year;
df_top1['month'] = df_top1['date'].dt.month;

df_top1_promo = df_top1[df_top1['Promo_flag'] == 1];





# =============================================================================
# 觀察整體營收情況
# =============================================================================
# 先按年月彙總
df_top1_monthly = (
    df_top1.groupby(['year','month'], as_index=False)['sales']
    .sum()
    .sort_values(['year','month'])
)

# 建立時間索引 (每月第一天)
df_top1_monthly['date'] = pd.to_datetime(
    df_top1_monthly['year'].astype(str) + '-' +
    df_top1_monthly['month'].astype(str) + '-01'
)
df_top1_monthly = df_top1_monthly.set_index('date');

# 時間序列分解
result = seasonal_decompose(df_top1_monthly['sales'],
                            model='additive', period=12);

# 畫圖
fig = result.plot();
fig.set_size_inches(12, 8);
plt.tight_layout();
plt.savefig("output/top1_Store_Sales_all_sales_seasonal_decompose.png", dpi=300, bbox_inches="tight");
plt.show();
plt.close();

# =============================================================================
# 這家店的整體營收逐年成長（Trend 上升）。
# 每年有明顯的銷售季節性（Seasonal），可以針對高峰月份規劃促銷活動。
# 部分波動無法預測（Residual），屬於隨機或特殊事件造成。
# =============================================================================





# =============================================================================
# 觀察促銷營收情況
# =============================================================================
# 先按年月彙總
df_top1_monthly = (
    df_top1_promo.groupby(['year','month'], as_index=False)['sales']
    .sum()
    .sort_values(['year','month'])
)

# 建立時間索引 (每月第一天)
df_top1_monthly['date'] = pd.to_datetime(
    df_top1_monthly['year'].astype(str) + '-' +
    df_top1_monthly['month'].astype(str) + '-01'
)
df_top1_monthly = df_top1_monthly.set_index('date');

# 時間序列分解
result = seasonal_decompose(df_top1_monthly['sales'],
                            model='additive', period=12);

# 畫圖
fig = result.plot();
fig.set_size_inches(12, 8);
plt.tight_layout();
plt.savefig("output/top1_Store_Sales_onprom_sales_seasonal_decompose.png", dpi=300, bbox_inches="tight");
plt.show();
plt.close();


# =============================================================================
# 促銷策略可能對整體趨勢有正面影響（Trend 上升）。
# 每年有明顯的銷售季節性（Seasonal），可以針對高峰月份規劃促銷活動。
# 部分波動無法預測（Residual），屬於隨機或特殊事件造成。
# =============================================================================




# =============================================================================
# 觀察非促銷營收情況
# =============================================================================
# 先按年月彙總
df_top1_no_promo = df_top1[df_top1['Promo_flag'] == 0];
df_top1_monthly = (
    df_top1_no_promo.groupby(['year','month'], as_index=False)['sales']
    .sum()
    .sort_values(['year','month'])
)

# 建立時間索引 (每月第一天)
df_top1_monthly['date'] = pd.to_datetime(
    df_top1_monthly['year'].astype(str) + '-' +
    df_top1_monthly['month'].astype(str) + '-01'
)
df_top1_monthly = df_top1_monthly.set_index('date');

# 時間序列分解
result = seasonal_decompose(df_top1_monthly['sales'],
                            model='additive', period=12);

# 畫圖
fig = result.plot();
fig.set_size_inches(12, 8);
plt.tight_layout();
plt.savefig("output/top1_Store_Sales__no_onprom_sales_seasonal_decompose.png", dpi=300, bbox_inches="tight");
plt.show();
plt.close();

# =============================================================================
# 市場習慣轉移：消費者越來越依賴促銷才購買，原價銷售的吸引力降低。
# 季節性（Seasonal依然存在，即使不促銷，消費者在某些月份仍有固定的購買習慣。
#     ->代表店家過度依賴促銷才能維持營收。長期來看，這會侵蝕品牌價值（客人只在打折時買）
# 部分波動無法預測（Residual），屬於隨機或特殊事件造成。
# =============================================================================


# =============================================================================
# 假設檢定驗證 T-test
# =============================================================================
from scipy.stats import ttest_ind;
sales_promo = df_top1[df_top1['Promo_flag'] == 1]['sales'];
sales_no_promo = df_top1[df_top1['Promo_flag'] == 0]['sales'];


# 基本描述統計
print("促銷期間平均銷售:", sales_promo.mean());
print("非促銷期間平均銷售:", sales_no_promo.mean());

t_stat, p_val = ttest_ind(sales_promo, sales_no_promo, equal_var=False)  # Welch's T-test

print("T統計量:", t_stat)
print("P值:", p_val)

if p_val < 0.05:
    print("差異統計顯著 → 促銷期間銷售顯著高於非促銷")
else:
    print("差異不顯著 → 促銷期間銷售沒有明顯提升")



# =============================================================================
# 促銷期間平均銷售: 2776.709240789766
# 非促銷期間平均銷售: 547.2407977214001
# T統計量: 63.99046767428273
# P值: 0.0
# 差異統計顯著 → 促銷期間銷售顯著高於非促銷
# 
# 結論:
# 1.促銷效果明顯
# 	a.促銷期間銷售額遠高於非促銷期間（平均約 5 倍），T 檢定顯示統計顯著。
# 	b.說明促銷策略能顯著拉抬銷售。
# 
# 2.季節性趨勢
# 	a.每年 12 月促銷銷售額最高 → 年末節慶（聖誕、跨年）為高峰期。
# 	b.其他月份促銷效果雖高於非促銷，但沒有固定高峰 → 可視為促銷帶來的「即時刺激」。
# 
# 3.長期趨勢
# 	a.非促銷期間銷售趨勢略下降 → 自然銷售力不足，需要促銷或其他策略維持收入。
# 	b.促銷期間銷售呈上升或穩定 → 表示促銷能抵消自然下降趨勢。
# =============================================================================



















