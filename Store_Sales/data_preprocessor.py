# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:21:57 2025

@author: user
"""
import pandas as pd;
import numpy as np;
from sklearn.preprocessing import LabelEncoder;


class data_functions:

    def get_df_after_handled(self):
        df_train = self.merge_df();
        df_train = self.merge_holiday_columns(df_train);
        df_train = self.paser_date(df_train);
        df_train = self.fill_oil_price_na(df_train);
        df_train = self.add_weekend_feature(df_train);
        df_train = self.add_hot_day_feature(df_train,4,1.2);

        
        return df_train;
        

    def merge_df(self):
        
       df_train = pd.read_csv("train.csv", parse_dates=['date']);
       df_stores = pd.read_csv("stores.csv");
       df_transactions = pd.read_csv("transactions.csv", parse_dates=['date']);
       df_oil = pd.read_csv("oil.csv", parse_dates=['date']);

       df_train = df_train.merge(df_transactions, how="left",
                              on=["store_nbr", "date"]);
       df_train = df_train.merge(df_stores,how="left",on = "store_nbr");
       df_train = df_train.merge(df_oil,how = "left",on = "date");
       
       return df_train;

    def merge_holiday_columns(self,df_train):
        df_holidays = pd.read_csv("holidays_events.csv", parse_dates=['date']);

        df_holidays = df_holidays[df_holidays['transferred'] == False];

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
        return df_train;
    

    def paser_date(self,df_train):
        df_train['date'] = pd.to_datetime(df_train['date'])
        df_train['dayofweek'] = df_train['date'].dt.dayofweek  # 星期幾 (0=Mon)
        df_train['week'] = df_train['date'].dt.isocalendar().week
        df_train['month'] = df_train['date'].dt.month
        df_train['year'] = df_train['date'].dt.year
        df_train['day'] = df_train['date'].dt.day
        
        return df_train;
    
    def fill_oil_price_na(self,df_train):
        df_train = df_train.sort_values('date');
        df_train['dcoilwtico'] = df_train['dcoilwtico'].ffill();
        df_train['dcoilwtico'] = df_train['dcoilwtico'].bfill();
        
        return df_train;
    

    def add_weekend_feature(self,df_train):
        df_train['is_weekend'] = df_train['dayofweek'].isin([5, 6]);
        df_train['is_friday'] = df_train['dayofweek'] == 4;
        
        return df_train;


    def add_hot_day_feature(self, df, weekday, threshold):
        """
        根據指定 weekday 是否在該群體中為「熱賣日」來新增布林特徵欄位。
    
        參數:
            df: 資料表，需包含 'dayofweek', 'family', 'sales'
            weekday: 要檢查的星期幾（0=週一, 6=週日）
            threshold: 相對於該 family 全週平均的倍率門檻（如 1.3）
    
        回傳:
            新增 'day{weekday}_hot' 的 DataFrame
        """
        
        weekday_map = {
            0: "monday",
            1: "tuesday",
            2: "wednesday",
            3: "thursday",
            4: "friday",
            5: "saturday",
            6: "sunday"
        }
    
        day_name = weekday_map.get(weekday, f"day{weekday}");
        new_col = f"is_{day_name}_hot";

    
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

    def add_lag_rolling_features(self,df, group_keys, target_cols, lags=[1,7,14], rolling_windows=[7,14]):
        df = df.copy()
        g = df.groupby(group_keys)
        
        # 加 lag 特徵
        for col in target_cols:
            for lag in lags:
                new_col = f'{col}_lag_{lag}'
                df[new_col] = g[col].shift(lag)
        
        # 加 rolling mean/std 特徵
        for col in target_cols:
            for window in rolling_windows:
                roll_mean_col = f'{col}_roll_mean_{window}'
                roll_std_col = f'{col}_roll_std_{window}'
                
                df[roll_mean_col] = g[col].shift(1).rolling(window).mean()
                df[roll_std_col] = g[col].shift(1).rolling(window).std()
        

        
        return df;
    
    def remove_nan_in_lag(self,df):
                

        # 先列出所有 lag/rolling 欄位
        cols_to_check = [f"sales_lag_{lag}" for lag in [1,7,14]] + \
                        [f"transactions_lag_{lag}" for lag in [1,7,14]] + \
                        [f"sales_roll_mean_{w}" for w in [7,14]] + \
                        [f"sales_roll_std_{w}" for w in [7,14]] + \
                        [f"transactions_roll_mean_{w}" for w in [7,14]] + \
                        [f"transactions_roll_std_{w}" for w in [7,14]]
        
        #處理nan欄位
        df = df.dropna(subset=cols_to_check);
        
        return df;

    
    def add_day_category(self, df):
        df = df.copy();
    
        df["day_category"] = "week";
    
        df.loc[df["holiday_type"].notna(), "day_category"] = "holiday";
    
        df.loc[
            (df["dayofweek"].isin([5, 6])) & (df["holiday_type"].isna()),
            "day_category"
        ] = "weekend"
    
        return df;

    def add_annual_growth_rate(self,df):
        df = df.copy();
        
        annual_sales = (
            df.groupby(["store_nbr","family","year","month","day_category"])["sales"]
            .sum()
            .reset_index()
            .sort_values(["store_nbr", "family", "year", "month", "day_category"])
        )
        
        annual_sales['sales_last_year'] = annual_sales.groupby(["store_nbr", "family", "month", "day_category"])['sales'].shift(1);
        
        annual_sales['annual_growth_rate'] = (
            (annual_sales['sales'] - annual_sales['sales_last_year']) / 
            annual_sales['sales_last_year'].replace(0, np.nan)  # 避免除以0
        )

        annual_sales['annual_growth_rate'] = annual_sales['annual_growth_rate'].fillna(0);
        
        df = df.merge(
            annual_sales[["store_nbr", "family", "year", "month", "day_category", "annual_growth_rate"]],
            on=["store_nbr", "family", "year", "month", "day_category"],
            how="left"
        )
        
        return df;
        
        
        
    def label_encode_columns(self,df, cols):
        df = df.copy()
        for col in cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        return df
        
    
    #簡單的利用關鍵字將節日分為 國慶 地方跟宗教
    def create_multi_label_holiday_features(self,df):
        # 建立三個二元欄位，預設為0
        df["is_national_holiday"] = 0;
        df["is_religious_holiday"] = 0;
        df["is_local_holiday"] = 0;
    
        # 標註國定假日：holiday_type 或 locale 是 National
        df.loc[(df["holiday_type"] == "National") | (df["locale"] == "National"), "is_national_holiday"] = 1;
    
        # 標註地方假日：holiday_type 或 locale 是 Regional 或 Local
        df.loc[(df["holiday_type"].isin(["Regional", "Local"])) | (df["locale"].isin(["Regional", "Local"])), "is_local_holiday"] = 1;
    
        # 標註宗教假日：可以用 description 關鍵字判斷（你可以根據資料自己補充）
        religious_keywords = ["Viernes Santo", "Semana Santa", "Carnaval", "Navidad", "Easter", "Christmas"];  # 範例關鍵字
        df.loc[df["description"].str.contains('|'.join(religious_keywords), case=False, na=False), "is_religious_holiday"] = 1;
    
        return df;
        
        
    #利用sentence transformer來將節日分類
# =============================================================================
#     def labeling_holiday_by_sen_trans(self,df):
# =============================================================================
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        