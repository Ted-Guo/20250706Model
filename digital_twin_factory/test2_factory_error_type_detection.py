import pandas as pd;
import random;
import joblib;

from generate_test2_factory_error_type_csv import FactorySimulator;

model = joblib.load("test2_factory_error_type_model.pkl");


factory =  FactorySimulator();
# 模擬幾筆資料
sample_data = factory.simulate(total_samples=20);



# 轉成 DataFrame
df_sample = pd.DataFrame(sample_data);

# 特徵欄位
feature_cols = ["motor_temp", "motor_speed", "motor_current",
                "conveyor_speed", "conveyor_load",
                "cooling_flow", "cooling_temp"];

# 輸入模型
X_input = df_sample[feature_cols];
Y_pred = model.predict(X_input);

# 原本標籤
label_cols = ["motor_err", "conveyor_err", "cooling_err"];
Y_true = df_sample[label_cols].values;

# 印出比對結果
for i, (pred, true) in enumerate(zip(Y_pred, Y_true)):
    print(f"樣本 {i+1}:");
    print(f"  預測 -> {pred}");
    print(f"  正解 -> {true}");
    print(f"  是否完全正確? -> {all(pred == true)}\n");