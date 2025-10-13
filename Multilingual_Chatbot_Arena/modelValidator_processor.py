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
    def kfold_auc(self, X, y, seed=42, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed);
        aucs = [];
    
        scaler = StandardScaler(with_mean=False) if issparse(X) else StandardScaler();
        pipe = Pipeline([
            ("scaler", scaler),
            ("clf", LogisticRegression(max_iter=2000))
        ])
    
        for tr_idx, va_idx in skf.split(X, y):
            # 避免 validation fold 單一類別
            if len(set(y[va_idx])) < 2:
                continue;
            pipe.fit(X[tr_idx], y[tr_idx])
            p = pipe.predict_proba(X[va_idx])[:, 1];
            aucs.append(roc_auc_score(y[va_idx], p));
    
        return np.array(aucs);


    def ablation_study_test(self, df, base_cols, test_cols, target_col="winner", seeds=[42, 52, 62], n_splits=5):
        """
        消融測試 (多 seed)
        """
        y = (df[target_col] == "model_a").astype(int).values;
        X_base = df[base_cols].values;
        X_plus = df[base_cols + test_cols].values;
    
        auc_base_all, auc_plus_all = [], [];
    
        for seed in seeds:
            auc_base = self.kfold_auc(X_base, y, seed=seed, n_splits=n_splits);
            auc_plus = self.kfold_auc(X_plus, y, seed=seed, n_splits=n_splits);
    
            auc_base_all.extend(auc_base);
            auc_plus_all.extend(auc_plus);
    
            print(f"Seed {seed} | AUC base mean={auc_base.mean():.4f}, AUC+test mean={auc_plus.mean():.4f}, Δ={auc_plus.mean()-auc_base.mean():.4f}");
    
        auc_base_all = np.array(auc_base_all);
        auc_plus_all = np.array(auc_plus_all);
    
        delta = auc_plus_all - auc_base_all;
    
        print("\n=== Overall Result ===");
        print("AUC base    : mean =", auc_base_all.mean(), "±", auc_base_all.std());
        print("AUC + test  : mean =", auc_plus_all.mean(), "±", auc_plus_all.std());
        print("ΔAUC per fold =", delta);
        print("ΔAUC mean =", delta.mean());
    
        # 成對檢定
        print("Paired t-test :", ttest_rel(auc_plus_all, auc_base_all));
        print("Wilcoxon test :", wilcoxon(auc_plus_all, auc_base_all));
    
        return auc_base_all, auc_plus_all;