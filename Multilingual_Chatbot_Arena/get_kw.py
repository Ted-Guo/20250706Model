# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 00:45:43 2025

@author: user
"""

import pandas as pd;
import numpy as np;

from polt1 import plot_data;
from data_pre_processoer import data_pre_funs;
from key_feature import FeatureExtractor;

plot_funs = plot_data();
pre_funs = data_pre_funs();

df_train = pd.read_parquet("pre_datas_0822.parquet");

#抽取關鍵詞+算覆蓋率
fe = FeatureExtractor();
df_train = fe.add_kw_coverage_features(df_train);

#加入情感特徵,直接把 response_a / response_b 的情感分數加到 df_train
sentiment_a = df_train['response_a'].apply(fe.analyze_sentiments).apply(pd.Series);
sentiment_a = sentiment_a.add_prefix("a_");  # a_neg, a_neu, a_pos, a_compound

sentiment_b = df_train['response_b'].apply(fe.analyze_sentiments).apply(pd.Series);
sentiment_b = sentiment_b.add_prefix("b_");

sentiment_prompt = df_train['prompt'].apply(fe.analyze_sentiments).apply(pd.Series);
sentiment_prompt = sentiment_prompt.add_prefix("prompt_");
df_train = pd.concat([df_train, sentiment_a, sentiment_b, sentiment_prompt], axis=1);


out_path = "pre_datas_0903.parquet";
df_train.to_parquet(out_path, index=False);
import os;
print(f"Save done! File exists? {os.path.exists(out_path)}");
print("Current working dir:", os.getcwd());