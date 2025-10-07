import pandas as pd;
from sklearn.model_selection import train_test_split;
from sklearn.multioutput import MultiOutputClassifier;
from xgboost import XGBClassifier;
from sklearn.metrics import classification_report;
import joblib;


df = pd.read_csv("test2_factory_error_type_train_data.csv");

# =============================================================================
# 特徵欄位
# =============================================================================
feature_cols = [
    "motor_temp", "motor_speed", "motor_current",
    "conveyor_speed", "conveyor_load",
    "cooling_flow", "cooling_temp"
];

# =============================================================================
# 多輸出標籤
# =============================================================================
label_cols = ["motor_err", "conveyor_err", "cooling_err"];

X = df[feature_cols];
Y = df[label_cols];

# -----------------------------
# 分割訓練/驗證集
# -----------------------------
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
);

# -----------------------------
# 建立 MultiOutput XGBoost
# -----------------------------
base_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False
);

multi_model = MultiOutputClassifier(base_xgb);
multi_model.fit(X_train, Y_train);

# -----------------------------
# 驗證與報告
# -----------------------------
Y_pred = multi_model.predict(X_val);

for i, label in enumerate(label_cols):
    print(f"--- {label} --- \n");
    print(classification_report(Y_val[label], Y_pred[:, i]));

# -----------------------------
# 模型儲存
# -----------------------------
joblib.dump(multi_model, "test2_factory_error_type_model.pkl");
print("模型已儲存：test2_factory_error_type_model.pkl \n");
