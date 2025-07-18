# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 10:09:05 2025


[0]	train-rmse:1039.13872	val-rmse:1033.90243
[50]	train-rmse:454.08052	val-rmse:445.56665
[100]	train-rmse:397.33665	val-rmse:390.00644
[150]	train-rmse:365.65453	val-rmse:361.03657
[200]	train-rmse:346.19775	val-rmse:343.96983
[250]	train-rmse:331.93956	val-rmse:330.87886
[300]	train-rmse:320.49660	val-rmse:321.99337
[350]	train-rmse:311.90367	val-rmse:315.07574
[400]	train-rmse:304.23584	val-rmse:308.57494
[450]	train-rmse:298.68688	val-rmse:304.90207
[500]	train-rmse:294.08753	val-rmse:301.55897
[550]	train-rmse:288.29678	val-rmse:296.63834
[600]	train-rmse:282.94562	val-rmse:292.68795
[650]	train-rmse:277.71242	val-rmse:289.05873
[700]	train-rmse:274.56889	val-rmse:286.63987
[750]	train-rmse:271.62176	val-rmse:284.40140
[800]	train-rmse:268.62956	val-rmse:282.12307
[850]	train-rmse:264.77236	val-rmse:280.01789
[900]	train-rmse:260.87814	val-rmse:277.03058
[950]	train-rmse:258.34561	val-rmse:275.23949
[999]	train-rmse:256.10606	val-rmse:273.81900


@author: user
"""

import pandas as pd;
import numpy as np;
import xgboost as xgb;

from sklearn.model_selection import train_test_split;


# =============================================================================
# family:商品種類 date:日期 sales:交易金額(0代表沒賣出去) onpromotion:促銷商品數量 store_nbr:店號
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

df_train = df_train.merge(df_stores,how="left",on = "store_nbr");
df_train = df_train.merge(df_oil,how = "left",on = "date");

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
df_train.rename(columns={
    'type_x': 'store_type',
    'type_y': 'holiday_type'
}, inplace=True);


#print(df_train.isnull().sum().sort_values(ascending=False));

# =============================================================================
# 確認資料缺失原因
# =============================================================================
# 篩選出 description 缺失的資料
missing_holiday = df_train[df_train['description'].isna()];

#description nan時,locale holiday type也是nan -> 應該非假期,也非補假 ->平日
#df = missing_holiday.head();
# 看這些資料的日期中，有沒有在 holidays 原始資料出現過
missing_dates = missing_holiday['date'].unique();

# 和原始 holiday 資料的日期交集比較
missing_dates_in_holidays = df_holidays[df_holidays['date'].isin(missing_dates)];

# =============================================================================
# print(f"缺失 description 的日期共有: {len(missing_dates)} 個");
# print(f"這些日期中，在 holidays.csv 中實際有出現的有: {len(missing_dates_in_holidays)} 個");
# =============================================================================

df_train['is_holiday'] = df_train['holiday_type'].notna().astype(int);


# =============================================================================
# 解決油價缺失欄位->前後補值法(油價變動相對平緩)
# =============================================================================
df_train = df_train.sort_values('date');
df_train['dcoilwtico'] = df_train['dcoilwtico'].ffill();
df_train['dcoilwtico'] = df_train['dcoilwtico'].bfill();


#print(df_train[['sales', 'onpromotion', 'dcoilwtico']].describe());

df_train['date'] = pd.to_datetime(df_train['date'])
df_train['dayofweek'] = df_train['date'].dt.dayofweek  # 星期幾 (0=Mon)
df_train['week'] = df_train['date'].dt.isocalendar().week
df_train['month'] = df_train['date'].dt.month
df_train['year'] = df_train['date'].dt.year
df_train['day'] = df_train['date'].dt.day
df_train['is_weekend'] = df_train['dayofweek'].isin([5, 6]);



#通常周末消費力高 5=Sat, 6=Sun,額外新增判斷是否為周五
df_train.groupby('dayofweek')['sales'].mean().plot(kind='bar');
#特定用品在周五有明顯增長...
df_train.groupby(['dayofweek', 'family'])['sales'].mean().unstack().plot(figsize=(16,6));

#新增feature
df_train['is_weekend'] = df_train['dayofweek'].isin([5, 6]);
df_train['is_friday'] = df_train['dayofweek'] == 4;

#新增一個feature來判斷該商品是否是週五熱銷商品
def add_hot_day_feature(df, weekday, threshold):
    """
    根據指定 weekday 是否在該群體中為「熱賣日」來新增布林特徵欄位。

    參數:
        df: 資料表，需包含 'dayofweek', 'family', 'sales'
        weekday: 要檢查的星期幾（0=週一, 6=週日）
        threshold: 相對於該 family 全週平均的倍率門檻（如 1.3）

    回傳:
        新增 'day{weekday}_hot' 的 DataFrame
    """
    new_col = f"day{weekday}_hot";

    # 計算 family × dayofweek 的平均銷售
    day_avg = df.groupby(['family', 'dayofweek'])['sales'].mean().reset_index();
    family_avg = df.groupby('family')['sales'].mean().reset_index();

    # 過濾指定 weekday 的平均值
    target = day_avg[day_avg['dayofweek'] == weekday][['family', 'sales']];
    target = target.rename(columns={'sales': 'target_avg'});
    family_avg = family_avg.rename(columns={'sales': 'weekly_avg'});

    # 合併 & 計算比例
    merged = pd.merge(target, family_avg, on='family');
    merged['ratio'] = merged['target_avg'] / merged['weekly_avg'];

    # 判斷是否熱賣
    hot_families = merged[merged['ratio'] > threshold]['family'].tolist();

    # 新增欄位
    df[new_col] = ((df['dayofweek'] == weekday) & (df['family'].isin(hot_families))).astype(int);
    return df;

df_train = add_hot_day_feature(df_train,4,1.3);

def preprocess_for_xgb(df, category_cols=None):
    """
    將指定的欄位轉成 pandas 的 category 型別，方便給 XGBoost 使用。
    
    Parameters:
        df (pd.DataFrame): 要處理的資料集
        category_cols (list, optional): 欲轉換的類別欄位名稱清單
                                        預設會使用推薦欄位集。
    
    Returns:
        pd.DataFrame: 轉換後的資料集
    """
    if category_cols is None:
        category_cols = [
            'store_nbr', 'family', 'state', 'city', 'type',
            'cluster', 'holiday_type'
        ]
        
    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype("category");
    
    return df

df_train = preprocess_for_xgb(df_train);






# 資料清理函數
def clean_for_xgb(df):
    df = df.copy();
    
    # 移除無法轉為數值的欄位
    drop_cols = ['date', 'description', 'store_type', 'locale'];
    df = df.drop(columns=[col for col in drop_cols if col in df.columns]);

    # 類別欄位轉為數值
    for col in df.select_dtypes(include='category').columns:
        df[col] = df[col].cat.codes;
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype("category").cat.codes;

    return df;

# 特徵與目標變數拆分
X = df_train.drop(columns=["sales"]);
y = df_train["sales"];

# 切分資料集（隨機抽樣，涵蓋全年節慶）
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 清理訓練與驗證資料
X_train_clean = clean_for_xgb(X_train);
X_val_clean = clean_for_xgb(X_val);

# 建立 DMatrix
dtrain = xgb.DMatrix(X_train_clean, label=y_train);
dval = xgb.DMatrix(X_val_clean, label=y_val);

# 模型參數
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.1,
    "max_depth": 6,
    "seed": 42
}

# 訓練模型
evals = [(dtrain, "train"), (dval, "val")]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50
)

# 預測
dtest = xgb.DMatrix(X_val_clean);
y_pred = model.predict(dtest);

