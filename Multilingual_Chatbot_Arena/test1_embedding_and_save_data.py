# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 13:58:26 2025

@author: user
"""

import pandas as pd;
import numpy as np;

from polt1 import plot_data;
from data_pre_processoer import data_pre_funs;
from key_feature import FeatureExtractor;

plot_funs = plot_data();
pre_funs = data_pre_funs();

df_train = pd.read_parquet("train.parquet");
df = df_train.iloc[0];
#plot_funs.plot_by_group(df_raw);

#檢查數量最多語言前20名是否有資料不平衡
#plot_funs.get_top20_AB_Proportion(df_raw);
#檢查prompt是問句與非問句之間數量
#plot_funs.get_data_is_question(df_train);




#clear text
for col in ['prompt', 'response_a', 'response_b']:
    df_train[col] = df_train[col].apply(pre_funs.clean_text);
    


print(f" embedding for isn't question target before merge data count:{df_train.shape[0]}");

# =============================================================================
# embedding for isn't question target
# =============================================================================

#將prompt embedding
df_prompt_embed = pre_funs.embedding_data(df_train, "prompt", "emb_prompt");
#將responseA embedding
df_a_embed      = pre_funs.embedding_data(df_train, "response_a", "emb_response_a");
#將responseB embedding
df_b_embed      = pre_funs.embedding_data(df_train, "response_b", "emb_response_b");

#合併回原始 df（靠 id 對齊）
df_train = pre_funs.merge_embeddings(df_train, [df_prompt_embed, df_a_embed, df_b_embed]);

print(f"embedding for isn't question target after merge data count:{df_train.shape[0]}");
print("===========================================================================");
# cosine similarity
df_train = pre_funs.get_cos(df_train, "emb_prompt", "emb_response_a", pre_funs.embedding_dim, "cos_prompt_a");
df_train = pre_funs.get_cos(df_train, "emb_prompt", "emb_response_b", pre_funs.embedding_dim, "cos_prompt_b");


# =============================================================================
# embedding for question target
# =============================================================================

#將prompt,response_A combine + embedding
df_train['combine_a'] = df_train.apply(lambda row: pre_funs.combine_features(row['prompt'], row['response_a']), axis=1);
df_combine_a_embed = pre_funs.embedding_data(df_train, "combine_a", "emb_combine_a");

#將prompt,response_b combine + embedding
df_train['combine_b'] = df_train.apply(lambda row: pre_funs.combine_features(row['prompt'], row['response_b']), axis=1);
df_combine_b_embed = pre_funs.embedding_data(df_train, "combine_b", "emb_combine_b");

#合併回原始 df（靠 id 對齊）
df_train = pre_funs.merge_embeddings(df_train, [df_combine_a_embed, df_combine_b_embed]);

# cosine similarity
df_train = pre_funs.get_cos(df_train, "emb_combine_a", "emb_combine_b", pre_funs.embedding_dim, "cos_combine_a_b");
print(f"embedding for question target after merge data count:{df_train.shape[0]}");
print("===========================================================================");



#抽取關鍵詞+算覆蓋率
fe = FeatureExtractor();
df_train = fe.add_kw_coverage_features(df_train);

#加入情感特徵,直接把 response_a / response_b 的情感分數加到 df_train
sentiment_a = df_train['response_a'].apply(fe.analyze_sentiments).apply(pd.Series);
sentiment_a = sentiment_a.add_prefix("a_");  # a_neg, a_neu, a_pos, a_compound

sentiment_b = df_train['response_b'].apply(fe.analyze_sentiments).apply(pd.Series);
sentiment_b = sentiment_b.add_prefix("b_");

sentiment_prompt = df_train['prompt'].apply(fe.analyze_sentiments).apply(pd.Series);
sentiment_prompt = sentiment_b.add_prefix("prompt");
df_train = pd.concat([df_train, sentiment_a, sentiment_b, sentiment_prompt], axis=1);


out_path = "pre_datas_0903.parquet";
df_train.to_parquet(out_path, index=False);
import os;
print(f"Save done! File exists? {os.path.exists(out_path)}");
print("Current working dir:", os.getcwd());





