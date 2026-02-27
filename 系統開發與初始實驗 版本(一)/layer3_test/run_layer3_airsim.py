"""
run_layer3_airsim.py - Layer 3 Flight Controller (AirSim 整合版)
================================================================
本腳本將 Layer 3 的高頻控制迴圈 (50Hz) 與 AirSim API 串接。

特點：
  1. 閉環控制：讀取 AirSim 真實座標作為 PID 的回授位點。
  2. 即時避障：結合模擬感知（獲取障礙物座標）計算 APF 斥力。
  3. 物理驅動：透過 `moveByVelocityAsync` 將計算出的向量發送至 AirSim。
"""

import time
import airsim
from layer3_standalone import Layer3Controller, Vector3

class AirSimLayer3(Layer3Controller):
    """擴展 Layer 3 以支持 AirSim 物理讀寫"""
    def __init__(self, hz: int = 50):
        super().__init__(hz=hz)
        self.client = airsim.MultirotorClient()
        try:
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            print("🔗 Layer 3 已連接至 AirSim 並取得控制權")
        except Exception as e:
            print(f"❌ AirSim 連接或控制權獲取失敗: {e}")

    def update_state_from_airsim(self):
        """同步實體狀態到控制器"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        
        self.pos = Vector3(pos.x_val, pos.y_val, pos.z_val)
        # 這裡不直接更新 self.vel，因為 self.vel 是由控制器輸出的期望值

    def apply_velocity_to_airsim(self):
        """將控制器計算的速度向量發送至 AirSim"""
        # 注意：AirSim 的 moveByVelocityAsync 是持續執行的，
        # 我們以 50Hz (dt=0.02s) 的頻率更新，duration 設為 0.1s 以確保平滑
        self.client.moveByVelocityAsync(
            self.vel.x, 
            self.vel.y, 
            self.vel.z, 
            duration=0.1
        )

def main():
    # 1. 初始化 AirSim 版 Layer 3
    controller = AirSimLayer3(hz=50)
    
    # 2. 設定目標點 (NED 座標: x=50, y=50, z=-15)
    target = Vector3(50.0, 50.0, -15.0)
    controller.target = target
    
    # 3. 模擬一些障礙物 (現實中應由感知模組動態更新)
    controller.obstacles = [{"x": 25.0, "y": 25.0, "z": -15.0}]
    
    print(f"\n[Layer 3 AirSim Integration] 任務啟動：前往 {target}...")
    
    # 執行起飛
    print("🚀 起飛中...")
    controller.client.takeoffAsync().join()
    
    try:
        # 4. 高頻控制迴圈 (50Hz)
        dt = 1.0 / 50.0
        while True:
            start_loop = time.time()
            
            # 從 AirSim 讀取
            controller.update_state_from_airsim()
            
            # 計算 PID + APF
            controller.step()
            
            # 寫入 AirSim
            controller.apply_velocity_to_airsim()
            
            # 檢查是否抵達
            if controller.pos.dist(target) < 1.0:
                print("🏁 抵達目標點區域，進入懸停")
                controller.client.hoverAsync().join()
                break
                
            elapsed = time.time() - start_loop
            time.sleep(max(0, dt - elapsed))
            
    except KeyboardInterrupt:
        print("\n🛑 停止 Layer 3 控制")
        controller.client.hoverAsync().join()

if __name__ == "__main__":
    main()
