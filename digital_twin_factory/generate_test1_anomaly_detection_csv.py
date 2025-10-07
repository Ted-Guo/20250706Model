
import random
import csv

NUM_SAMPLES = 100
OUTPUT_FILE = "test1_anomaly_detection_data.csv"

# 異常判斷函數
def check_anomaly(motor_temp, cooling_flow, conveyor_load):
    if motor_temp > 80 or cooling_flow < 0.5 or conveyor_load > 50:
        return 1
    return 0


# 產生單筆模擬元件數據
def generate_sample():
    # 正常範圍
    motor_temp = random.uniform(40, 70)
    motor_speed = random.uniform(500, 1000)
    motor_current = random.uniform(1.5, 4.0)

    conveyor_load = random.uniform(10, 50)
    conveyor_speed = random.uniform(20, 60)

    cooling_flow = random.uniform(0.5, 2.0)
    cooling_temp = random.uniform(20, 35)

    # 隨機決定是否異常 (50% 機率)
    if random.random() < 0.5:
        anomaly_choice = random.choice(["motor", "cooling", "conveyor"])
        if anomaly_choice == "motor":
            motor_temp = random.uniform(81, 100)
        elif anomaly_choice == "cooling":
            cooling_flow = random.uniform(0.1, 0.49)
        elif anomaly_choice == "conveyor":
            conveyor_load = random.uniform(51, 60)

    # 判斷 y
    y = check_anomaly(motor_temp, cooling_flow, conveyor_load)


    return {
        "motor_temp": round(motor_temp, 2),
        "motor_speed": round(motor_speed, 2),
        "motor_current": round(motor_current, 2),
        "conveyor_speed": round(conveyor_speed, 2),
        "conveyor_load": round(conveyor_load, 2),
        "cooling_flow": round(cooling_flow, 2),
        "cooling_temp": round(cooling_temp, 2),
        "y": y
    }

# 產生資料集
data = [generate_sample() for _ in range(NUM_SAMPLES)]

# 存成 CSV
fieldnames = list(data[0].keys())
with open(OUTPUT_FILE, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(f"CSV 已產生: {OUTPUT_FILE}")
