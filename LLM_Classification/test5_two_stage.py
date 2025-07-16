"""
Created on Sun Jul 13 09:44:09 2025

=== 第一階段結果 (Tie vs Non-Tie) ===
              precision    recall  f1-score   support

     Non-Tie       0.71      0.96      0.82      7944
         Tie       0.59      0.13      0.21      3552

    accuracy                           0.70     11496
   macro avg       0.65      0.54      0.51     11496
weighted avg       0.67      0.70      0.63     11496

=== 第二階段結果 (A vs B) ===
              precision    recall  f1-score   support

           A       0.63      0.63      0.63      4013
           B       0.62      0.62      0.62      3931

    accuracy                           0.62      7944
   macro avg       0.62      0.62      0.62      7944
weighted avg       0.62      0.62      0.62      7944

=== 模擬完整預測（融合兩階段） ===
Overall Accuracy: 0.6632392087269691
              precision    recall  f1-score   support

           A       0.63      0.83      0.72     20064
           B       0.64      0.81      0.72     19652
         tie       0.90      0.31      0.46     17761

    accuracy                           0.66     57477
   macro avg       0.72      0.65      0.63     57477
weighted avg       0.72      0.66      0.64     57477

@author: user
"""


import pandas as pd;
import numpy as np;
import xgboost as xgb;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import classification_report, accuracy_score;

# === Step 1: 讀取資料 ===
df_raw = pd.read_parquet("features_0716.parquet");
df_raw = df_raw.drop(columns=['prompt'], errors='ignore');

# === Step 2: 建立標籤欄位 ===
label_map = {'A': 0, 'B': 1, 'tie': 2};
df_raw['label_num'] = df_raw['label'].map(label_map);
df_raw['is_tie'] = (df_raw['label'] == 'tie').astype(int);
df_raw['label_ab'] = df_raw['label'].map({'A': 0, 'B': 1});

# === Step 3: 預測是否為 tie ===
X1 = df_raw.drop(columns=['label', 'label_num', 'label_ab', 'is_tie'], errors='ignore');
y1 = df_raw['is_tie'];
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, stratify=y1, random_state=42);

clf1 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)
clf1.fit(X_train1, y_train1);
y_pred1 = clf1.predict(X_test1);

print("=== 第一階段結果 (Tie vs Non-Tie) ===");
print(classification_report(y_test1, y_pred1, target_names=['Non-Tie', 'Tie']));

# === Step 4:針對非 tie，預測 A,B ===
df_notie = df_raw[df_raw['is_tie'] == 0].copy();
X2 = df_notie.drop(columns=['label', 'label_num', 'is_tie', 'label_ab'], errors='ignore');
y2 = df_notie['label_ab'];
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, stratify=y2, random_state=42);

clf2 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)
clf2.fit(X_train2, y_train2);
y_pred2 = clf2.predict(X_test2);

print("=== 第二階段結果 (A,B) ===");
print(classification_report(y_test2, y_pred2, target_names=['A', 'B']));

# === Step 5: 模擬實際預測（融合兩階段） ===
X_all = df_raw.drop(columns=['label', 'label_num', 'label_ab', 'is_tie'], errors='ignore');
X_all_reset = X_all.reset_index(drop=True);
tie_preds = clf1.predict(X_all_reset);

final_preds = []
for i, tie in enumerate(tie_preds):
    if tie == 1:
        final_preds.append("tie");
    else:
        sample = X_all_reset.iloc[[i]];
        ab_pred = clf2.predict(sample)[0];
        final_preds.append("A" if ab_pred == 0 else "B");

# === Step 6: 評估整體預測效能 ===
y_true = df_raw['label'].values
print("=== 模擬完整預測（融合兩階段） ===");
print("Overall Accuracy:", accuracy_score(y_true, final_preds));
print(classification_report(y_true, final_preds, target_names=['A', 'B', 'tie']));