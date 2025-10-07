import random;
import time;
import json;
import joblib;
import pandas as pd;


class Motor:
    def __init__(self):
        self.temperature = 45;
        self.speed = 1000;
        self.current = 2.0;

    def update(self, load, cooling_flow):
        # 溫度隨負載上升，隨冷卻下降
        self.temperature += random.uniform(0.1, 0.5) * (load / 20) - random.uniform(0.2, 0.5) * cooling_flow;
        # 溫度回復機制
        if self.temperature > 90:
            self.temperature -= random.uniform(5, 10);
        self.temperature = max(40, min(self.temperature, 100));

        # 轉速隨負載下降
        self.speed = max(500, 1000 - load * random.uniform(1.5, 2.5));

        # 電流隨負載增加
        self.current = min(4.5, 1.5 + load * 0.05);

        return {
            "motor_temp": round(self.temperature, 2),
            "motor_speed": round(self.speed, 2),
            "motor_current": round(self.current, 2)
        };
    
    
class Conveyor:
    def __init__(self):
        self.speed = 50;
        self.load = 20;

    def update(self, motor_speed):
        # 傳送帶速度取決於馬達轉速
        self.speed = max(20, min(60, motor_speed / random.uniform(15, 25)));
        # 負載隨機波動
        self.load = max(10, min(50, self.load + random.randint(-3, 3)));
        return {
            "conveyor_speed": round(self.speed, 2),
            "conveyor_load": self.load
        };
    
    
class Cooling:
    def __init__(self):
        self.flow = 1.0;
        self.temperature = 25;

    def update(self):
        # 冷卻流量隨機波動
        self.flow = max(0.3, min(2.0, self.flow + random.uniform(-0.2, 0.2)));
        # 冷卻水溫隨環境變化
        self.temperature = max(20, min(35, self.temperature + random.uniform(-0.5, 0.5)));
        return {
            "cooling_flow": round(self.flow, 2),
            "cooling_temp": round(self.temperature, 2)
        };