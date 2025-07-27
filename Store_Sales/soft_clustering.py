# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 21:23:09 2025

@author: user
"""
import hdbscan;
import numpy as np;
import pandas as pd;
from sentence_transformers import SentenceTransformer;

class trans_locale_function:
    #先將節日轉成向量
    def description_sentence_trans(self, df):
        output_path="holiday_with_embeddings.parquet";
        # 1. 模型選擇（對節慶、跨語言較佳）
        model = SentenceTransformer('distiluse-base-multilingual-cased-v2');
    
        # 2. 確保 event_text 欄位存在，並補 NaN
        df["event_text"] = df["description"].fillna("no_event");
    
        # 3. 分批嵌入處理
        batch_size = 10000;
        total = len(df);
        embedding_list = [];
        index_list = [];
    
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total);
            print(f"處理第 {start} ~ {end} 筆...");
           
            batch_df = df.iloc[start:end];
            batch_texts = batch_df["event_text"].tolist();
            embeddings = model.encode(batch_texts, show_progress_bar=False).astype(np.float32);
           
            embedding_list.append(embeddings);
            index_list.extend(batch_df.index.tolist());
    
        # 4. 合併所有嵌入
        all_embeddings = np.vstack(embedding_list);
        emb_df = pd.DataFrame(
            all_embeddings,
            index=pd.Index(index_list, name="index"),
            columns=[f"embedding_{i}" for i in range(all_embeddings.shape[1])]
        );
    
        # 5. 合併回原始資料（index 自動對齊）
        df_with_embeddings = df.join(emb_df);
    
        # 6. 儲存為 Parquet（可壓縮, 支援欄位名）
        df_with_embeddings.to_parquet(output_path, index=False);
        print(f"已儲存到 {output_path}，總列數：{len(df_with_embeddings)}");
    
        return df_with_embeddings;
    
    def hdbscan_clusting(self, df):
        X = np.vstack(df['embedding'])
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2);
        df['cluster'] = clusterer.fit_predict(X);
        
        
        df.groupby('cluster')['description'].apply(list)
        return df;