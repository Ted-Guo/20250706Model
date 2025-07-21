# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:20:59 2025

Reloaded modules: data_preprocessor
[0]	train-rmse:841.58962	eval-rmse:847.93268
[20]	train-rmse:219.86361	eval-rmse:282.51289
[40]	train-rmse:198.85621	eval-rmse:273.34195
[60]	train-rmse:182.39917	eval-rmse:269.70834
[80]	train-rmse:169.81150	eval-rmse:267.51823
[100]	train-rmse:161.55049	eval-rmse:265.39298
[120]	train-rmse:155.37995	eval-rmse:263.95980
[140]	train-rmse:149.73320	eval-rmse:262.63686
[160]	train-rmse:145.10878	eval-rmse:261.95781
[180]	train-rmse:141.84131	eval-rmse:261.49895
[200]	train-rmse:138.70775	eval-rmse:261.01331
[220]	train-rmse:134.72353	eval-rmse:260.74313
[240]	train-rmse:132.50067	eval-rmse:260.49049
[260]	train-rmse:129.59582	eval-rmse:260.25772
[280]	train-rmse:127.22899	eval-rmse:260.12500
[300]	train-rmse:124.79875	eval-rmse:258.99228
[320]	train-rmse:122.69183	eval-rmse:258.80738
[340]	train-rmse:120.31400	eval-rmse:258.50648
[360]	train-rmse:118.27141	eval-rmse:258.37958
[380]	train-rmse:115.82891	eval-rmse:258.31272
[400]	train-rmse:114.23759	eval-rmse:258.17391
[420]	train-rmse:112.42632	eval-rmse:257.81282
[440]	train-rmse:111.11447	eval-rmse:257.90595
[460]	train-rmse:109.62774	eval-rmse:257.79964
[480]	train-rmse:108.15640	eval-rmse:257.78489
[500]	train-rmse:106.85030	eval-rmse:257.68143
[520]	train-rmse:105.80225	eval-rmse:257.54905
[540]	train-rmse:104.62156	eval-rmse:257.38203
[560]	train-rmse:103.51188	eval-rmse:257.38090
[580]	train-rmse:102.29751	eval-rmse:257.28763
[600]	train-rmse:101.32439	eval-rmse:257.17640
[620]	train-rmse:100.30758	eval-rmse:257.13924
[640]	train-rmse:99.20135	eval-rmse:257.08336
[660]	train-rmse:98.30340	eval-rmse:256.98885
[680]	train-rmse:97.45672	eval-rmse:256.89366
[700]	train-rmse:96.42134	eval-rmse:256.85337
[720]	train-rmse:95.66045	eval-rmse:256.84631
[740]	train-rmse:94.83030	eval-rmse:256.68570
[760]	train-rmse:93.91298	eval-rmse:256.62718
[780]	train-rmse:93.05439	eval-rmse:256.69203
[800]	train-rmse:92.28559	eval-rmse:256.63950
[820]	train-rmse:91.51029	eval-rmse:256.60713
[840]	train-rmse:90.77207	eval-rmse:256.67557
[860]	train-rmse:90.12178	eval-rmse:256.61040

@author: user
"""

import pandas as pd;
import numpy as np;
import xgboost as xgb

from data_preprocessor import data_functions;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import mean_squared_error;



common_functions = data_functions();

df_train = common_functions.get_df_after_handled();

#df_train.info();   # 顯示欄位名稱、型別與非空值數量

df_train = common_functions.add_lag_rolling_features(
    df=df_train,
    group_keys=['store_nbr', 'family'],
    target_cols=['sales', 'transactions'],
    lags=[1, 7, 14],
    rolling_windows=[7, 14]
);

df_train = common_functions.add_day_category(df_train);

df_train = df_train.drop(columns=["holiday_type", "description", "locale"]);

df_train = common_functions.add_annual_growth_rate(df_train);



categorical_cols = ['family', 'store_type', 'city', 'state', 'day_category'];
df_train = common_functions.label_encode_columns(df_train, categorical_cols);



df_train = common_functions.remove_nan_in_lag(df_train);


# 假設你的目標欄位是 'sales'
target = 'sales'

# 選擇訓練特徵（去除id、日期和目標欄位等）
drop_cols = ['id', 'date', target]
feature_cols = [col for col in df_train.columns if col not in drop_cols]

X = df_train[feature_cols]
y = df_train[target]

# 分割訓練與驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立 DMatrix (XGBoost 專用格式)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# 設定參數
params = {
    'objective': 'reg:squarederror',  # 迴歸問題
    'eval_metric': 'rmse',
    'seed': 42,
    'tree_method': 'hist'
}

# 訓練模型，並監控驗證集
evals = [(dtrain, 'train'), (dval, 'eval')]

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    early_stopping_rounds=50,
    evals=evals,
    verbose_eval=20
)

# 預測驗證集
y_pred = model.predict(dval)

# 評估
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f'Validation RMSE: {rmse}')





