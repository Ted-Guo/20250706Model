# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 16:19:07 2025

@author: user
"""
import pandas as pd;
import numpy as np;
import torch;

from sentence_transformers import SentenceTransformer;

class data_pre_funs:
    
    def __init__(self, model_name='sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens'):
        self.model = SentenceTransformer(model_name);
        self.embedding_dim = self.model.encode(["test"]).shape[1];
        
        
    def clean_text(self, text):
        if pd.isnull(text) or not isinstance(text, str) or text.strip() == "":
            return "";
        text = text.replace('\n', ' ').replace('\r', ' ').strip();
        return text;
        

    def embedding_data(self, df, target_column, column_name, batch_size = 32):
        
        embeddings = self.model.encode(
            df[target_column].tolist(),
            batch_size=batch_size,
            show_progress_bar=True
        );
        
        df_embed = pd.DataFrame(
            embeddings,
            columns=[f"{column_name}_{i}" for i in range(embeddings.shape[1])],
            index=df.index  # 保持和原始 df 對齊
        );
        # 保留 id
        df_embed['id'] = df['id'];
        
        return df_embed;
    
    
    def merge_embeddings(self, df, df_emb_list):
        """
        將多個 embedding DataFrame merge 回原始 df
        """
        for df_emb in df_emb_list:
            df = df.merge(df_emb, on='id', how='left');
        return df;
    

    def cos_features(self, u, v):
        torch_u = torch.tensor(u, dtype=torch.float32);
        torch_v = torch.tensor(v, dtype=torch.float32);
        
        u_norm = torch.nn.functional.normalize(torch_u, dim=1);
        v_norm = torch.nn.functional.normalize(torch_v, dim=1);
        
        return (u_norm * v_norm).sum(dim=1).cpu().numpy();


    def get_cos(self, df, col_prefix1, col_prefix2, embedding_dim, column_name):
        # 動態取欄位名稱
        cols1 = [f"{col_prefix1}_{i}" for i in range(embedding_dim)];
        cols2 = [f"{col_prefix2}_{i}" for i in range(embedding_dim)];
        
        # 檢查欄位是否存在
        for c in cols1 + cols2:
            if c not in df.columns:
                raise KeyError(f"缺少欄位: {c}");
        
        cos_np = self.cos_features(
            df[cols1].values,
            df[cols2].values
        );
        
        df[column_name] = cos_np;
        return df;