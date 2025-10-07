import pandas as pd;
import random;

components = {
    "Conveyor": [
        ("傳送帶負載過高", "物料過多或卡住", "減少負載，清理障礙物"),
        ("傳送帶速度異常波動", "馬達驅動不穩定或控制系統異常", "檢查馬達與控制系統"),
        ("傳送帶停止運行", "緊急停止或電源中斷", "檢查緊急開關與電源"),
        ("傳送帶偏移或跑偏", "導向輪鬆動或皮帶張力不足", "調整導向輪及皮帶張力"),
        ("傳送帶異常噪音", "滾輪磨損或物料卡住", "檢查滾輪及清理物料"),
    ],
    "Motor": [
        ("馬達溫度過高", "冷卻不足或負載過高", "降低負載，檢查冷卻系統"),
        ("馬達轉速不穩定", "控制器異常或電壓不穩定", "檢查控制器與電源"),
        ("馬達異常噪音", "軸承磨損或異物進入", "更換軸承，清理異物"),
        ("馬達電流過大", "負載過重或短路", "降低負載，檢查線路"),
        ("馬達停止運轉", "電源故障或保護器動作", "檢查電源與保護器"),
    ],
    "Cooling": [
        ("冷卻水流量不足", "水泵故障或管路阻塞", "檢查水泵及管路"),
        ("冷卻水溫過高", "環境溫度過高或冷卻系統效率下降", "降低環境溫度，檢查冷卻系統"),
        ("冷卻水漏水", "管路破裂或接頭鬆脫", "檢查管路並修復"),
        ("冷卻流量波動", "控制閥異常或水泵效率下降", "檢查控制閥與水泵"),
        ("冷卻系統異常噪音", "水泵軸承磨損或氣泡", "更換軸承或排氣"),
    ]
};

severity_levels = ["高", "中", "低"];

data = [];

# 每個元件生成 50 條訊息
for component, templates in components.items():
    for i in range(50):
        template = random.choice(templates);
        error_msg = f"{template[0]} #{i+1}";
        possible_causes = template[1];
        recommended_action = template[2];
        severity = random.choice(severity_levels);
        
        data.append({
            "error_message": error_msg,
            "category": component,
            "possible_causes": possible_causes,
            "recommended_action": recommended_action,
            "severity": severity
        });

df = pd.DataFrame(data);
df.to_csv("test3_RAG.csv", index=False, encoding="utf-8-sig");
print("RAG 資料集已生成 (每個元件 50 條錯誤訊息)，CSV 檔名: test3_RAG.csv");
