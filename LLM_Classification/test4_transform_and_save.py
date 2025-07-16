import pandas as pd;
import numpy as np;
import torch;


from Data_preprocessor import Pre_sentence_trans;
from StyleFeatureExtractor import StyleFeatureExtractor;
from sentence_transformers import SentenceTransformer;
from sklearn.metrics.pairwise import cosine_similarity;
from sklearn.decomposition import PCA;
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer;


df_train = pd.read_csv("train.csv");
# 1. 語意特徵（句嵌入 + cosine + PCA）
pre_tran = Pre_sentence_trans();
data_sen_tran, y = pre_tran.fit(df_train);

# 2. 語言風格特徵（寫作風格等）
feature_ext = StyleFeatureExtractor();
df_style_a = feature_ext.extract(df_train, response_col='response_a', prefix='a_');
df_style_b = feature_ext.extract(df_train, response_col='response_b', prefix='b_');

# 3. 合併語意 + 風格特徵
df_combined = pd.merge(data_sen_tran, df_style_a, on='id', how='inner');
df_combined = pd.merge(df_combined, df_style_b, on='id', how='inner');

#4.加入情緒特徵
analyzer = SentimentIntensityAnalyzer();
def analyze_sentiments(text):
    return analyzer.polarity_scores(text);


# 5.分別處理 response_a 與 response_b 的情感分數
sentiment_a_scores = df_train['response_a'].apply(analyze_sentiments).apply(pd.Series).add_prefix("a_");
sentiment_b_scores = df_train['response_b'].apply(analyze_sentiments).apply(pd.Series).add_prefix("b_");


# 6.合併 VADER 特徵
df_sent = pd.concat([df_train[['id']], sentiment_a_scores, sentiment_b_scores], axis=1);
df_combined = pd.merge(df_combined, df_sent, on='id', how='inner');


# 7. 移除不需要的欄位（原始 prompt / response / label 等）
#df_combined = df_combined.drop(columns=['prompt', 'response_a', 'response_b'], errors='ignore');

# 8. 輸出為 parquet 格式
df_combined = df_combined.drop(columns=['model_a','model_b','response_a', 'response_b'], errors='ignore');
df_combined.to_parquet("features_0716.parquet", index=False);








