import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
import os;


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


# 每年每月分促銷/非促銷彙總
df_top1_monthly = (
    df_top1
    .groupby(['year','month','Promo_flag'], as_index=False)['sales']
    .sum()
);

#取得所有年
years = sorted(df_top1_monthly['year'].unique());

# 逐年繪圖
os.makedirs("output", exist_ok=True);
for yr in years:
    temp = df_top1_monthly[df_top1_monthly['year'] == yr];
    if temp.empty:
        continue;

    plt.figure(figsize=(8,5))
    ax = sns.barplot(
        data=temp,
        x='month',
        y='sales',
        hue='Promo_flag',
        palette='Set2'
    );

    plt.title(f"Store {top1_id} - Monthly Sales ({yr})");
    plt.xlabel("Month");
    plt.ylabel("Total Sales");
    plt.legend(title="Promotion", labels=["No Promo", "Promo"]);

    for p in ax.patches:
        height = p.get_height();
        ax.annotate(f"{height:,.0f}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9);

    plt.tight_layout();
    plt.savefig(f"output/top1_Store_Sales_{yr}.png", dpi=300, bbox_inches="tight");
    plt.close();








# =============================================================================
# 1. 促銷策略的效果明顯
#    a.促銷上線後，雖然第一個月沒有立即反應，但從第二個月起，「促銷銷售額幾乎每個月都超過非促銷銷售額」，顯示策略需要一個「延遲期 (lag effect)」 才能發揮。
#    b.促銷在 12 月尤其強勢，每年都是該年度的高峰，顯示促銷和「年末需求」（例如聖誕節、新年）有疊加效果。
# 
# 2. 非促銷沒有穩定的季節高峰
#    a.沒促銷的情況下，12 月銷售額不一定是最高，說明「自然需求並沒有明顯的年末旺季」，促銷才是推高 12 月銷售的主要因素。
#    b.其他月份的非促銷銷售呈現較平緩、不固定的起伏，顯示它受到「一般市場需求」影響，而非固定的季節性。
# 
# 3. 策略面啟示
#    a.促銷帶來的銷售提升「不只是短期現象」，而是持續且可預測的。
#    b.年末檔期（尤其 12 月）若「疊加促銷」可以達到最大收益。
#    c.可以利用這個特性，在年末之外的月份「嘗試不同促銷強度或組合」，觀察是否也能創造類似的差距。
# 
# 結論:
#     a.促銷策略是主要的營收驅動因素，帶來明顯的延遲效應與持續優勢，而非促銷的自然需求並不具備穩定的季節高峰。
#     b.促銷策略有明顯的「滯後效應」
# =============================================================================
