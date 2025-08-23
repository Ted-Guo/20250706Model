# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 01:51:47 2025

@author: user
"""
import numpy as np;
import pandas as pd;
from sklearn.model_selection import StratifiedKFold;
from sklearn.preprocessing import StandardScaler;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.linear_model import LogisticRegression;
from sklearn.metrics import roc_auc_score, f1_score, log_loss;
from scipy.stats import ttest_rel, wilcoxon;
from scipy.sparse import issparse;


class ModelValidator:
    def ablation_study_test(self, df, base_cols, test_cols, target_col="winner"):
        """
        消融測試
        df: pandas dataframe
        base_cols_arys: list, 基礎特徵欄位
        test_cols_arys: list, 要加入測試的特徵欄位
        target_col: 目標欄位名稱 (預設 winner)
        """

        # 將 target 轉成二元 0/1
        y = (df[target_col] == "model_a").astype(int).values

        X_base = df[base_cols].values
        X_plus = df[base_cols + test_cols].values

        # 計算 K-fold AUC
        auc_base = self.kfold_auc(X_base, y)
        auc_plus = self.kfold_auc(X_plus, y)

        print("AUC base    :", auc_base, "mean =", auc_base.mean())
        print("AUC + test  :", auc_plus, "mean =", auc_plus.mean())
        print("ΔAUC (+test - base) =", (auc_plus - auc_base).mean())

        # 成對檢定 (paired t-test / Wilcoxon)
        print("Paired t-test :", ttest_rel(auc_plus, auc_base))
        print("Wilcoxon test :", wilcoxon(auc_plus, auc_base))

        return auc_base, auc_plus
    
    def kfold_auc(self, X, y, seed=42, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed);
        aucs = [];

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False) if issparse(X) else StandardScaler()),  # dense matrix 預設即可
            ("clf", LogisticRegression(max_iter=2000))
        ]);

        for tr_idx, va_idx in skf.split(X, y):
            pipe.fit(X[tr_idx], y[tr_idx]);
            p = pipe.predict_proba(X[va_idx])[:, 1];
            aucs.append(roc_auc_score(y[va_idx], p));

        return np.array(aucs);