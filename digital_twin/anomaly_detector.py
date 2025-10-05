import pandas as pd;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.metrics import accuracy_score;
import joblib;
from sklearn.model_selection import train_test_split;


# -----------------------------
# 1. 讀取 CSV 訓練資料
# -----------------------------
df = pd.read_csv("training_data.csv");

# 特徵欄位（去掉 y）
feature_cols = [col for col in df.columns if col != "y"];
X = df[feature_cols];
y = df["y"];

# -----------------------------
# 2. 訓練 Random Forest
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42);

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
);

model.fit(X_train, y_train);



# -----------------------------
# 3. 計算訓練資料正確率
# -----------------------------
y_pred_test = model.predict(X_test);
acc_test = accuracy_score(y_test, y_pred_test);
print(f"測試資料準確率: {acc_test*100:.2f}%");

# -----------------------------
# 4. 保存模型
# -----------------------------
joblib.dump(model, "anomaly_model.pkl");
print("模型已保存為 anomaly_model.pkl");

# -----------------------------
# 5. 定義函數：判斷單筆資料是否異常
# -----------------------------
def predict_anomaly(sample_dict):
    """ 0=正常, 1=異常"""
    df_sample = pd.DataFrame([sample_dict]);
    pred = model.predict(df_sample)[0];
    return pred;

# -----------------------------
# 6. 範例測試
# -----------------------------
if __name__ == "__main__":
    # 範例單筆資料（可從 simulator 取得）
    sample = {
        "motor_temp": 85.2,
        "motor_speed": 920,
        "motor_current": 3.2,
        "conveyor_speed": 40,
        "conveyor_load": 45,
        "cooling_flow": 0.7,
        "cooling_temp": 25,
    };

    y_pred = predict_anomaly(sample);
    print("異常判斷結果:", y_pred);

