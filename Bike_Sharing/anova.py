# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 14:39:58 2025

@author: user
"""
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;

from scipy.stats import f;
from tabulate import tabulate;
import statsmodels.api as sm;
from statsmodels.formula.api import ols;

class anova_fun:
    def get_anova_data(self, df):
        # 準備資料
        groups = [df[df['weekday'] == i]['count'] for i in range(7)];
        # 計算總樣本數與群數
        n_groups = len(groups);          # k
        n_total = sum([len(g) for g in groups]);  # N
        # 整體平均
        grand_mean = df['count'].mean();
        # 組間平方和 (SS_between)
        ss_between = sum([len(g) * (g.mean() - grand_mean) ** 2 for g in groups]);
        # 組內平方和 (SS_within)
        ss_within = sum([sum((g - g.mean()) ** 2) for g in groups]);
        # 自由度
        df_between = n_groups - 1;
        df_within = n_total - n_groups;
        # 平均平方
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        # F 值
        F = ms_between / ms_within
        # p 值
        p_value = 1 - f.cdf(F, df_between, df_within);

        # 輸出 ANOVA 表
        anova_table = pd.DataFrame({
            'Source': ['Between Groups', 'Within Groups'],
            'SS': [ss_between, ss_within],
            'df': [df_between, df_within],
            'MS': [ms_between, ms_within],
        })

        anova_table['F'] = [F, ''];
        anova_table['p-value'] = [p_value, ''];
        print(tabulate(anova_table, headers='keys', tablefmt='github'));
        return;
        
    def two_way_anova(self,df):
        model = ols('count ~ C(weekday) + C(hour) + C(weekday):C(hour)', data=df).fit();
        anova_table = sm.stats.anova_lm(model, typ=2);
        print(anova_table);
        return;
        
    def hitmap(self, df, target='count'):
        numeric_df = df.select_dtypes(include=['int', 'float']);
        corr = numeric_df.corr();
    
        plt.figure(figsize=(12, 10));
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0);
        plt.title("Correlation Matrix");
        plt.show();
    
        print("\nTop correlated with target:");
        print(corr[target].sort_values(ascending=False));
        return;

    def hitmap_weekday_hour(self,df):
        
       pivot_table = df.groupby(['weekday', 'hour'])['count'].mean().unstack();
    
       plt.figure(figsize=(10, 6));
       sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu");
       plt.title("Average Rental Count (hour vs weekday)");
       plt.xlabel("Weekday");
       plt.ylabel("Hour");
       plt.show();

       return;

    def plot_box(self,df,x,x_label,y,y_label,title):
        #week箱型圖
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'];
        plt.rcParams['axes.unicode_minus'] = False;
        plt.figure(figsize=(10,6));
        sns.boxplot(data=df, x=x, y=y);
        plt.title(title);
        plt.xlabel(x_label);
        plt.ylabel(y_label);
        return;
        
        
    def plot_week_chart(self,df):
        #長條圖
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'];
        plt.rcParams['axes.unicode_minus'] = False;
        df.groupby('weekday')['count'].mean().plot(kind='bar')
        plt.title("平均租借量（星期別）")
        plt.xlabel("星期（0=一，6=日）")
        plt.ylabel("平均 count")
        plt.show()
        print(df.groupby('weekday')['count'].mean());
        return;
        
        
    def plot_hour(self,df):

        plt.figure(figsize=(12, 6));
        sns.boxplot(x='hour', y='count', data=df);
        plt.title('Count Distribution by Hour');
        plt.xlabel('Hour of Day');
        plt.ylabel('Rental Count');
        plt.show();

        return;
        
    def plot_hour_group_weekday(self,df):
        # 計算 weekday + hour 的平均與標準差
        df_group = df.groupby(['weekday', 'hour'])['count'].agg(['mean', 'std']).reset_index()
        df_group.columns = ['weekday', 'hour', 'mean_count', 'std_count']
    
        # 畫出 hour 對 count 的 boxplot（單變數分析）
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='hour', y='count', data=df)
        plt.title('Count Distribution by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Rental Count')
        plt.show()
    
        # 畫出 weekday vs hour 的 heatmap（雙變數交互）
        pivot_table = df_group.pivot(index='hour', columns='weekday', values='mean_count')
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu")
        plt.title("Average Rental Count (hour vs weekday)")
        plt.xlabel("Weekday (0=Mon ~ 6=Sun)")
        plt.ylabel("Hour")
        plt.show()
        return;
        
    
    def add_weekday_hour_feature(self,df):
        df['weekday_hour'] = df['weekday'].astype(str) + "_" + df['hour'].astype(str)
        return df;