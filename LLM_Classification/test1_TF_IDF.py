# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 10:00:43 2025

              precision    recall  f1-score   support

          A勝       0.42      0.52      0.46      4030
          B勝       0.43      0.49      0.46      3929
          平手       0.30      0.16      0.21      3537

    accuracy                           0.40     11496
   macro avg       0.38      0.39      0.38     11496
weighted avg       0.38      0.40      0.38     11496

@author: user
"""
import pandas as pd;
import numpy as np;
import re;
import nltk;

from nltk.tokenize import word_tokenize;
from nltk.corpus import stopwords;
from nltk.stem import WordNetLemmatizer;
from sklearn.feature_extraction.text import TfidfVectorizer;
from sklearn.linear_model import LogisticRegression;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import classification_report;


from nltk.data import find;
nltk.download('punkt', download_dir='');


# =============================================================================
# read data and remove model_a,model_b feature
# =============================================================================
df_train = pd.read_csv("train.csv");
df_train.drop(columns=['model_a', 'model_b'], inplace=True);


# =============================================================================
# Check for data imbalance
# =============================================================================
num_tie = df_train['winner_tie'].sum();
print(f"平手的筆數：{num_tie}");
num_a = df_train['winner_model_a'].sum();
print(f"a的筆數：{num_a}");
num_b = df_train['winner_model_b'].sum();
print(f"b的筆數：{num_b}");


# =============================================================================
# Removing punctuation is not useful for prediction
# =============================================================================
def remove_Punctuation(text):
    return re.sub(r'[^\w\s]', '', text);


for col in ["prompt","response_a","response_b"]:
    df_train[col] = df_train[col].apply(remove_Punctuation);
   
# =============================================================================
# Convert uppercase to lowercase
# =============================================================================
df_train["prompt"] = df_train["prompt"].str.lower();
df_train["response_a"] = df_train["response_a"].str.lower();
df_train["response_b"] = df_train["response_b"].str.lower();


# =============================================================================
# Tokenization
nltk.download('stopwords');
stop_words = set(stopwords.words('english'));

def remove_stopwords(text):
    tokens = word_tokenize(text);
    filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
    return ' '.join(filtered_tokens)

for col in ['prompt', 'response_a', 'response_b']:
    df_train[col] = df_train[col].apply(remove_stopwords);



lemmatizer = WordNetLemmatizer();
def lemmatize_text(text):
    tokens = word_tokenize(text);
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens];
    return ' '.join(lemmatized);

for col in ['prompt', 'response_a', 'response_b']:
    df_train[col] = df_train[col].apply(lemmatize_text);
    
# =============================================================================
# TF-IDF
# =============================================================================

#pairwise
df_train['input_a'] = df_train['prompt'] + ' ' + df_train['response_a'];
df_train['input_b'] = df_train['prompt'] + ' ' + df_train['response_b'];
#create corpus
corpus = pd.concat([df_train['input_a'], df_train['input_b']], ignore_index=True);
vectorizer = TfidfVectorizer(max_features=1000);
vectorizer.fit(corpus);
X_a = vectorizer.transform(df_train['input_a']);
X_b = vectorizer.transform(df_train['input_b']);
#print(vectorizer.get_feature_names());

def get_label(row):
    if row['winner_model_a'] == 1:
        return 0
    elif row['winner_model_b'] == 1:
        return 1
    else:
        return 2

y = df_train.apply(get_label, axis=1).values;
X_diff = X_a - X_b;



# =============================================================================
# test LogisticRegression
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X_diff, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['A勝', 'B勝', '平手']));

