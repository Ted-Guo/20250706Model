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