import random;
import pandas as pd;
from FactoryComponent import Motor, Conveyor, Cooling;

# -----------------------------
# 主模擬器：整合各元件
# -----------------------------
class FactorySimulator:
    def __init__(self):
        self.motor = Motor();
        self.conveyor = Conveyor();
        self.cooling = Cooling();

        # 7種有效異常組合 (排除全0)
        self.error_combinations = [
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1,1,0],
            [1,0,1],
            [0,1,1],
            [1,1,1]
        ];

    def simulate(self, total_samples=5000):
        data = [];
        for _ in range(total_samples):
            self.cooling.update();
            # 隨機選一組異常
            motor_err, conveyor_err, cooling_err = random.choice(self.error_combinations);

            # 設定負載與速度
            motor_load = random.uniform(0.4, 0.8);
            self.motor.load = motor_load;

            # 模擬異常狀態
            if motor_err:
                self.motor.temperature += random.uniform(10, 30);
                if motor_err and motor_load < 0.9:
                    self.motor.load = random.uniform(0.9, 1.2);

            if cooling_err:
                self.cooling.flow = random.uniform(0.0, 0.3);

            if conveyor_err:
                self.conveyor.speed += random.uniform(-80, 80);

            # 更新元件
            motor_state = self.motor.update(self.conveyor.load, self.cooling.flow);
            conveyor_state = self.conveyor.update(self.motor.speed);
            cooling_state = self.cooling.update();

            # 存入資料
            data.append({
                "motor_temp": round(motor_state["motor_temp"], 2),
                "motor_speed": round(motor_state["motor_speed"], 2),
                "motor_current": round(motor_state["motor_current"], 2),
                "conveyor_speed": round(conveyor_state["conveyor_speed"], 2),
                "conveyor_load": round(conveyor_state["conveyor_load"], 2),
                "cooling_flow": round(cooling_state["cooling_flow"], 2),
                "cooling_temp": round(cooling_state["cooling_temp"], 2),
                "motor_err": motor_err,
                "conveyor_err": conveyor_err,
                "cooling_err": cooling_err
            });

        return data;

# -----------------------------
# 主程式
# -----------------------------
if __name__ == "__main__":
    sim = FactorySimulator();
    data = sim.simulate(total_samples=5000);
    df = pd.DataFrame(data);
    df.to_csv("test2_factory_error_type_train_data.csv", index=False);

    print("資料集已產生完成");
    print(f"test2_factory_error_type_train_data.csv 筆數: {len(df)}");
    print("\n多標籤分布（每個元件異常比率）：");
    print(df[["motor_err", "conveyor_err", "cooling_err"]].mean().round(2));

    print("\n各組合分布：");
    print(df.groupby(["motor_err","conveyor_err","cooling_err"]).size());
