# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 23:38:12 2025
           precision    recall  f1-score   support

           A       0.45      0.44      0.44      4030
           B       0.46      0.44      0.45      3929
         tie       0.40      0.42      0.41      3537

    accuracy                           0.43     11496
   macro avg       0.43      0.43      0.43     11496
weighted avg       0.44      0.43      0.43     11496

@author: user
"""
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sentence_transformers import SentenceTransformer;
from sentence_trancefomer_preprocessor import BertPreprocessor;


import nltk;
from nltk.tokenize import word_tokenize;
from nltk import pos_tag;
from nltk.corpus import wordnet;
from nltk.stem import WordNetLemmatizer;



# =============================================================================
# use sentencetrainformer and save
# =============================================================================
# =============================================================================
# df_train = pd.read_csv("train.csv");
# df_train.drop(columns=['model_a', 'model_b'], inplace=True);
# 
# preprocessor = BertPreprocessor();
# X_a, X_b, X_diff, y = preprocessor.transform(df_train);
# np.savez('embedding_data.npz', X_a=X_a, X_b=X_b, X_diff=X_diff, y=y);
# =============================================================================
data = np.load('embedding_data.npz',allow_pickle=True);
X_a = data['X_a'];
X_b = data['X_b'];
X_diff = data['X_diff'];
X_mul = X_a * X_b;
X_concat = np.concatenate([X_a, X_b, X_diff], axis=1);
X_concat_mul = np.concatenate([X_a, X_b, X_diff, X_mul], axis=1);

y = data['y'];



# =============================================================================
# test rog
# =============================================================================
from sklearn.linear_model import LogisticRegression;
from sklearn.metrics import classification_report;
from sklearn.model_selection import train_test_split;
#X_train, X_test, y_train, y_test = train_test_split(X_diff, y, test_size=0.2, random_state=42);
X_train, X_test, y_train, y_test = train_test_split(X_concat, y, test_size=0.2, random_state=42);
X_train, X_test, y_train, y_test = train_test_split(X_concat_mul, y, test_size=0.2, random_state=42);

#clf = LogisticRegression(max_iter=1000);
clf = LogisticRegression(max_iter=1000, class_weight='balanced');

clf.fit(X_train, y_train);

y_pred = clf.predict(X_test);
print(classification_report(y_test, y_pred, target_names=['A', 'B', 'tie']));
# =============================================================================
# label_counts = pd.Series(y).value_counts();
# 
# label_counts.plot(kind='bar');
# plt.title("Label Distribution");
# plt.xlabel("Label");
# plt.ylabel("Count");
# plt.show();
# 
# =============================================================================
