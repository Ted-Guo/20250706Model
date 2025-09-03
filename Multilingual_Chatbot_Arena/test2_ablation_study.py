# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 01:51:04 2025

@author: user
"""

import pandas as pd;
import numpy as np;

from modelValidator_processor import ModelValidator;
from data_pre_processoer import data_pre_funs;

validator_funs = ModelValidator();
pre_funs = data_pre_funs();

df_train = pd.read_parquet("pre_datas_0822.parquet");



# 取得消融測試欄位  針對非問答類型 測試---cos_prompt_a
# =============================================================================
# AUC base    : [0.56060792 0.56621538 0.56283247 0.57204699 0.56756913] mean = 0.5658543753905377
# AUC + test  : [0.56102902 0.56627393 0.56276753 0.57310621 0.5671027 ] mean = 0.5660558780630304
# ΔAUC (+test - base) = 0.000201502672492615
# Paired t-test : TtestResult(statistic=np.float64(0.7839129054783908), pvalue=np.float64(0.47691003170874885), df=np.int64(4))
# Wilcoxon test : WilcoxonResult(statistic=np.float64(6.0), pvalue=np.float64(0.8125))
# =============================================================================

# =============================================================================
# df_no_q = df_train[~df_train['prompt'].str.contains(r"[?？]", na=False)];
# base_cols = [col for col in df_no_q.columns if col.startswith("emb_response_a") or col.startswith("emb_response_b") or col.startswith("emb_prompt")];
# test_cols = [col for col in df_no_q.columns if col.startswith("cos_prompt_a")];
# validator_funs.ablation_study_test(df_no_q, base_cols, test_cols);
# =============================================================================


# =============================================================================
# # 取得消融測試欄位  針對非問答類型 測試---cos_prompt_b
# AUC base    : [0.56060792 0.56621538 0.56283247 0.57204699 0.56756913] mean = 0.5658543753905377
# AUC + test  : [0.56097047 0.56670313 0.56342011 0.5719339  0.56832446] mean = 0.5662704140748117
# ΔAUC (+test - base) = 0.0004160386842739783
# Paired t-test : TtestResult(statistic=np.float64(2.82900915642204), pvalue=np.float64(0.04739266217390961), df=np.int64(4))
# Wilcoxon test : WilcoxonResult(statistic=np.float64(1.0), pvalue=np.float64(0.125))
# =============================================================================

# =============================================================================
# base_cols = [col for col in df_no_q.columns if col.startswith("emb_response_a") or col.startswith("emb_response_b") or col.startswith("emb_prompt")];
# test_cols = [col for col in df_no_q.columns if col.startswith("cos_prompt_b")];
# 
# validator_funs.ablation_study_test(df_no_q, base_cols, test_cols);
# =============================================================================


#cross encoder
# =============================================================================
# cross_score_a:
# AUC base    : [0.56060792 0.56621538 0.56283247 0.57204699 0.56756913] mean = 0.5658543753905377
# AUC + test  : [0.56156176 0.56625022 0.56300508 0.57258942 0.56815189] mean = 0.5663116736536746
# ΔAUC (+test - base) = 0.00045729826313680724
# Paired t-test : TtestResult(statistic=np.float64(2.811679742589963), pvalue=np.float64(0.04823442387800346), df=np.int64(4))
# Wilcoxon test : WilcoxonResult(statistic=np.float64(0.0), pvalue=np.float64(0.0625))
# =============================================================================


# =============================================================================
# cross_score_b: 提升幅度小但顯著 
# Seed 42 | AUC base mean=0.5659, AUC+test mean=0.5670, Δ=0.0012
# Seed 52 | AUC base mean=0.5619, AUC+test mean=0.5630, Δ=0.0011
# Seed 62 | AUC base mean=0.5684, AUC+test mean=0.5695, Δ=0.0011
# 
# === Overall Result ===
# AUC base    : mean = 0.5653779251621962 ± 0.005212982133855345
# AUC + test  : mean = 0.5664916147603901 ± 0.0052775111831613476
# ΔAUC per fold = [0.00143147 0.00160854 0.00081192 0.00177503 0.00026883 0.00117964
#  0.00063475 0.00141253 0.00118589 0.00100612 0.00094087 0.00146642
#  0.00119037 0.00139412 0.00039884]
# ΔAUC mean = 0.001113689598194112
# Paired t-test : TtestResult(statistic=np.float64(9.862601600840522), pvalue=np.float64(1.1084557504882247e-07), df=np.int64(14))
# Wilcoxon test : WilcoxonResult(statistic=np.float64(0.0), pvalue=np.float64(6.103515625e-05))
# =============================================================================


# =============================================================================
# cross_diff: 提升幅度小但顯著 
# 
# Seed 42 | AUC base mean=0.5659, AUC+test mean=0.5696, Δ=0.0038
# Seed 52 | AUC base mean=0.5619, AUC+test mean=0.5656, Δ=0.0037
# Seed 62 | AUC base mean=0.5684, AUC+test mean=0.5722, Δ=0.0038
# 
# === Overall Result ===
# AUC base    : mean = 0.5653779251621962 ± 0.005212982133855345
# AUC + test  : mean = 0.5691380662527289 ± 0.005114499422601635
# ΔAUC per fold = [0.00595025 0.00364882 0.00249724 0.00471325 0.00195621 0.00233952
#  0.00443045 0.00440432 0.0028757  0.00434856 0.00420759 0.00504473
#  0.00283944 0.00490953 0.0022365 ]
# ΔAUC mean = 0.0037601410905327387
# Paired t-test : TtestResult(statistic=np.float64(11.913875786859784), pvalue=np.float64(1.0258795747214775e-08), df=np.int64(14))
# Wilcoxon test : WilcoxonResult(statistic=np.float64(0.0), pvalue=np.float64(6.103515625e-05))
# =============================================================================

# =============================================================================
# pairs_a = list(zip(df_train["prompt"], df_train["response_a"]));
# pairs_b = list(zip(df_train["prompt"], df_train["response_b"]));
# df_train = pre_funs.cross_encoder(df_train, pairs_a, pairs_b);
# df_no_q = df_train[~df_train['prompt'].str.contains(r"[?？]", na=False)];
# 
# 
# base_cols = [col for col in df_no_q.columns if col.startswith("emb_response_a") or col.startswith("emb_response_b") or col.startswith("emb_prompt")];
# #test_cols = [col for col in df_no_q.columns if col.startswith("cross_score_a")];
# #test_cols = [col for col in df_no_q.columns if col.startswith("cross_score_b")];
# test_cols = [col for col in df_no_q.columns if col.startswith("cross_diff")];
# 
# validator_funs.ablation_study_test(df_no_q, base_cols, test_cols);
# =============================================================================




# =============================================================================
# cos_combine_a_b: 特徵無效
# Seed 42 | AUC base mean=0.5832, AUC+test mean=0.5833, Δ=0.0001
# Seed 52 | AUC base mean=0.5743, AUC+test mean=0.5743, Δ=-0.0001
# Seed 62 | AUC base mean=0.5795, AUC+test mean=0.5795, Δ=0.0000
# 
# === Overall Result ===
# AUC base    : mean = 0.5790208244490267 ± 0.00818827322171204
# AUC + test  : mean = 0.5790155625247634 ± 0.00829927091038816
# ΔAUC per fold = [ 1.74973840e-04 -3.62581080e-04  1.53665315e-04  2.50013554e-04
#   1.04287422e-04 -9.93934240e-05  5.83582881e-05 -2.28598738e-04
#  -4.55825817e-05 -1.27424035e-04  2.65739363e-04  5.07613512e-05
#  -9.63429727e-05 -2.62099845e-04  8.52946793e-05]
# ΔAUC mean = -5.261924263370391e-06
# Paired t-test : TtestResult(statistic=np.float64(-0.10773337795680224), pvalue=np.float64(0.9157359184326801), df=np.int64(14))
# Wilcoxon test : WilcoxonResult(statistic=np.float64(59.0), pvalue=np.float64(0.97796630859375))
# =============================================================================

df_q = df_train[df_train['prompt'].str.contains(r"[?？]", na=False)];


base_cols = [col for col in df_q.columns if col.startswith("emb_response_a") or col.startswith("emb_response_b") or col.startswith("emb_prompt")];
test_cols = [col for col in df_q.columns if col.startswith("cos_combine_a_b")];

validator_funs.ablation_study_test(df_q, base_cols, test_cols);












