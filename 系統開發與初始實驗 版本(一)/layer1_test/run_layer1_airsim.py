"""
run_layer1_airsim.py - Layer 1 Brain (AirSim 整合版)
======================================================
本腳本串接 AirSim API，從模擬器獲取即時座標與狀態，
並輸入 Layer 1 RAG 大腦進行決策。

特點：
  1. 使用 AirSim API 獲取真實座標 (Pos) 與區域 (Zone)。
  2. 呼叫 Layer 1 進行語意決策。
  3. 顯示決策結果，但不直接驅動物理（驅動由 Layer 3 負責）。
"""

import time
import airsim
from layer1_standalone import Layer1Brain

def get_zone_from_pos(x, y):
    """根據座標簡易判斷區域（模擬論文中的語意地圖解析）"""
    if 0 <= x <= 50 and 0 <= y <= 50:
        return "RT_FOR_01"  # Forest
    elif 50 < x <= 100 and 0 <= y <= 100:
        return "RT_OPE_01"  # Open Field
    elif x < 0 or y < 0:
        return "RT_BUI_01"  # Building Zone
    return "RT_VEG_01"      # Vegetation

def main():
    # 1. 初始化 AirSim 客戶端
    client = airsim.MultirotorClient()
    try:
        client.confirmConnection()
        print("🔗 已連接到 AirSim 模擬器")
    except Exception as e:
        print(f"❌ 無法連接 AirSim: {e}")
        return

    # 2. 初始化 Layer 1 Brain (使用免費組件)
    # 預設使用 Ollama 或本地 Mock
    brain = Layer1Brain(db_dir="db", use_ollama=True, verbose=True)

    print("\n[Layer 1 AirSim Integration] 啟動監控...")
    
    try:
        while True:
            # 3. 從 AirSim 獲取即時狀態
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            
            x, y, z = pos.x_val, pos.y_val, pos.z_val
            zone = get_zone_from_pos(x, y)
            
            print(f"\n📍 目前位置: ({x:.1f}, {y:.1f}, {z:.1f}) | 區域: {zone}")
            
            # 4. 模擬一個隨機任務需求（或可改為手動輸入）
            task = "在當前區域進行偵巡，搜尋任何紅色車輛目標"
            
            # 5. 呼叫大腦進行決策
            decision = brain.decide(
                task_instruction=task,
                current_state={"Zone": zone, "Alt": z},
                target_area=zone,
                battery_pct=100.0 # 假設滿電
            )
            
            print(f"💡 大腦決策: {decision['command']} | 理由: {decision['reasoning'][:100]}...")
            
            # 每 10 秒進行一次大腦推論（模仿高層級長週期決策）
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n🛑 停止 Layer 1 監控")

if __name__ == "__main__":
    main()
