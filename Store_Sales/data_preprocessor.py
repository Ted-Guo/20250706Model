# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:21:57 2025

@author: user
"""
import pandas as pd;
import numpy as np;


class data_functions:


    def merge_df(self):
        
       df_train = pd.read_csv("train.csv", parse_dates=['date']);
       df_stores = pd.read_csv("stores.csv");
       df_transactions = pd.read_csv("transactions.csv", parse_dates=['date']);
       df_holidays = pd.read_csv("holidays_events.csv", parse_dates=['date']);
       df_oil = pd.read_csv("oil.csv", parse_dates=['date']);

       df_train = df_train.merge(df_stores,how="left",on = "store_nbr");
       df_train = df_train.merge(df_oil,how = "left",on = "date");
       
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


    def preprocess_for_xgb(self, df, category_cols=None):
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
        
        return df;

    # 資料清理函數
    def clean_for_xgb(self, df):
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

    def classify_day_type(self, df):
        df = df.copy()
        df["day_type"] = "weekday"
        df.loc[df["date"].dt.weekday >= 5, "day_type"] = "weekend"  # 六日
        df.loc[df["holiday_type"].notna(), "day_type"] = "holiday"  # 國定假日
        return df


    # 將原始資料根據 day_type, month, year 計算平均銷售
    def compute_monthly_avg(self, df):
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
    
        return (
            df.groupby(["day_type", "month", "year"])["sales"]
            .mean()
            .reset_index()
        )

    
    # 計算每個 (day_type, month) 組別的平均年成長率
    def compute_growth_factors(self, historical_avg):
        def compute_avg_growth(group):
            group = group.sort_values("year")
            group["yearly_growth"] = group["sales"].pct_change()
            mean_growth = group["yearly_growth"].mean()
            return pd.Series({
                "avg_growth_rate": mean_growth if pd.notnull(mean_growth) else 0
            })
    
        return (
            historical_avg.groupby(["day_type", "month"])
            .apply(compute_avg_growth)
            .reset_index()
        )
