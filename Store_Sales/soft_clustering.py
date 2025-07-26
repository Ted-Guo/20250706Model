# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 21:23:09 2025

@author: user
"""
import hdbscan;
import numpy as np;
from sentence_transformers import SentenceTransformer;

class trans_locale_function:
    #先將節日轉成向量
    def description_sentence_trans(self, df):
        model = SentenceTransformer('all-MiniLM-L6-v2');
        df['embedding'] = df['description'].apply(lambda x: model.encode(str(x)));
        return df;
    
    def hdbscan_clusting(self, df):
        X = np.vstack(df['embedding'])
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2);
        df['cluster'] = clusterer.fit_predict(X);
        
        
        df.groupby('cluster')['description'].apply(list)
        return df;