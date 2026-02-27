"""
run_layer2_airsim.py - Layer 2 Behavior Planner (AirSim 整合版)
================================================================
本腳本將 Layer 2 的行為規劃與監控邏輯與 AirSim 客戶端串接。

特點：
  1. 即時同步：透過 10Hz 監控迴圈持續從 AirSim 獲取真實座標。
  2. 安全攔截：LLM 產生的航點若超出 AirSim 地圖自定義邊界，將立即攔截。
  3. 狀態反應：若 AirSim 中的無人機遇到異常（如通訊中斷模擬），切換至自主模式。
"""

import time
import airsim
from layer2_standalone import Layer2BehaviorPlanner, Waypoint, FSMState

class AirSimLayer2(Layer2BehaviorPlanner):
    """擴展 Layer 2 以支持 AirSim 資料獲取"""
    def __init__(self, env_path: str = "environment.json"):
        super().__init__(env_path=env_path)
        self.client = airsim.MultirotorClient()
        try:
            self.client.confirmConnection()
            print("🔗 Layer 2 已連接至 AirSim")
        except Exception as e:
            print(f"❌ AirSim 連接失敗: {e}")

    def update_drone_pos_from_airsim(self):
        """從 AirSim 更新內部座標狀態"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        self.simulate_drone_move(pos.x_val, pos.y_val, pos.z_val)

def main():
    # 1. 初始化 AirSim 版 Layer 2
    planner = AirSimLayer2(env_path="environment.json")
    
    print("\n[Layer 2 AirSim Integration] 啟動監控與座標同步...")
    
    # 2. 定義一個來自 Layer 1 的測試決策 (含一個禁航區航點與一個合法航點)
    test_decision = {
        "mission_id": "AIRSIM_TEST_001",
        "command": "PATROL",
        "target_area": "Area1",
        "waypoints": [
            {"x": 10.0, "y": 10.0, "z": -15.0, "label": "Normal_WP"},
            {"x": -50.0, "y": -50.0, "z": -15.0, "label": "Hacker_WP"} # 假設這是禁航區
        ],
        "altitude": -15.0,
        "speed": 5.0,
        "timeout_sec": 30.0
    }

    try:
        # 3. 啟動座標同步執行緒 (簡易演示，整合進 monitor 效果更好)
        # 這裡手動處理一次
        planner.update_drone_pos_from_airsim()
        
        # 4. 處理決策
        print(f"\n--- 嘗試下發 Layer 1 決策 ---")
        success = planner.process_layer1_decision(test_decision)
        
        if not success:
            print("🛡️  [驗證成功] Layer 2 正確攔截了含有非法座標的決策。")
        
        # 5. 修改為合法決策重試
        valid_decision = test_decision.copy()
        valid_decision["waypoints"] = [{"x": 10.0, "y": 10.0, "z": -15.0}]
        
        print(f"\n--- 嘗試下發合法決策 ---")
        if planner.process_layer1_decision(valid_decision):
            print("✅ 任務啟動成功")
            
            # 模擬運行 5 秒，持續同步位置
            for _ in range(50):
                planner.update_drone_pos_from_airsim()
                status = planner.get_status()
                if _ % 10 == 0:
                    print(f"📡 監控中 | 狀態: {status['fsm_state']} | 位置: {status['drone_pos']}")
                time.sleep(0.1)
        
        planner.stop()
        
    except KeyboardInterrupt:
        planner.stop()

if __name__ == "__main__":
    main()
