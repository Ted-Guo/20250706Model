# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 14:09:34 2025

@author: user
"""

class plot_data:
    
    def plot_by_group(self,df):
        print(df["language"].value_counts().head(20));


    def get_top20_AB_Proportion(self,df):
        top20_langs = df["language"].value_counts().head(20).index.tolist();
        
        df_top20 = df[df["language"].isin(top20_langs)];
        
        lang_stats_top20 = df_top20.groupby(["language", "winner"]).size().unstack(fill_value=0);
        
        lang_ratio_top20 = lang_stats_top20.div(lang_stats_top20.sum(axis=1), axis=0);
        print("==================================");
        print(lang_stats_top20);
        print("==================================");
        print(lang_ratio_top20);


    def get_data_is_question(self,df):
        print(f"總筆數: {df.shape[0]}");
        df["is_question"] = df["prompt"].str.strip().str.endswith("?");
        df_is_not_question = df[~df["is_question"]];
        print(f"非問句筆數: {df_is_not_question.shape[0]}");
        print(df[~df["is_question"]]["prompt"].head(20));
