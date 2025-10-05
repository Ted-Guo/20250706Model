import random;
import time;
import json;
import joblib;
import pandas as pd;

from FactoryComponent import Motor;
from FactoryComponent import Conveyor;
from FactoryComponent import Cooling;


# ---- 模擬主程式 ----
motor = Motor();
conveyor = Conveyor();
cooling = Cooling();

#資料更新區間
UPDATE_INTERVAL = 1;

model = joblib.load("anomaly_model.pkl");

def predict_anomaly(sample_dict):
    df_sample = pd.DataFrame([sample_dict]);
    return model.predict(df_sample)[0];

for step in range(20):
    cooling_state = cooling.update();
    motor_state = motor.update(conveyor.load, cooling_state["cooling_flow"]);
    conveyor_state = conveyor.update(motor.speed);

    # 隨機產生短暫異常事件
    if random.random() < 0.2:  # 20% 機率出現異常
        motor_state["motor_temp"] += random.uniform(10, 20);
        conveyor_state["conveyor_load"] += random.randint(10, 20);
        cooling_state["cooling_flow"] -= random.uniform(0.5, 1.0);

    system_state = {
        **motor_state,
        **conveyor_state,
        **cooling_state
    };
    
    y_pred = predict_anomaly(system_state);

    print("系統狀態:", json.dumps(system_state));
    print("模型判斷異常:", y_pred);
    print("="*50);
    time.sleep(UPDATE_INTERVAL);
