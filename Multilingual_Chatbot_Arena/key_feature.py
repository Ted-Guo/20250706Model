# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:06:45 2025

@author: user
"""

from sentence_transformers import SentenceTransformer;
from keybert import KeyBERT;
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer;
import pandas as pd;


class FeatureExtractor:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        # 初始化 multilingual sentence-transformer
        self.mbert = SentenceTransformer(model_name);
        self.kw_model = KeyBERT(model=self.mbert);
        self.analyzer = SentimentIntensityAnalyzer();

    def extract_keywords_multilingual(self, text, topn=5):
        """抽取多語言關鍵詞"""
        keywords = self.kw_model.extract_keywords(text, top_n=topn, stop_words=None);
        return [kw for kw, score in keywords];

    def keyword_coverage(self, prompt, response, topn=5):
        """計算 prompt 關鍵詞在 response 中的覆蓋率"""
        kws = set(self.extract_keywords_multilingual(prompt, topn=topn));
        if not kws:
            return 1.0;  # 沒關鍵詞就給 1
        resp_text = response.lower();
        return len([kw for kw in kws if kw in resp_text]) / len(kws);

    def add_kw_coverage_features(self, df, prompt_col="prompt", resp_a_col="response_a", resp_b_col="response_b", topn=5):
        """對 dataframe 增加關鍵詞覆蓋率特徵"""
        df["kw_cov_a"] = df.apply(lambda x: self.keyword_coverage(x[prompt_col], x[resp_a_col], topn=topn), axis=1);
        df["kw_cov_b"] = df.apply(lambda x: self.keyword_coverage(x[prompt_col], x[resp_b_col], topn=topn), axis=1);
        df["kw_cov_diff"] = df["kw_cov_a"] - df["kw_cov_b"];
        return df;

    def analyze_sentiments(self, text):
        return self.analyzer.polarity_scores(text);