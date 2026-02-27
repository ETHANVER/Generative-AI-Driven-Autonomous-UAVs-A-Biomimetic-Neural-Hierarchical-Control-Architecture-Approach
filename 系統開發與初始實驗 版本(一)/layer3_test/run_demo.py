"""
run_demo.py - Layer 3 避障展示 Demo
====================================
展示無人機在執行直航任務時，遇到障礙物如何透過 APF 斥力自動繞行。
"""

import time
import os
from layer3_standalone import Layer3Controller, Vector3
from mock_perception import MockPerception

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_demo_scene():
    controller = Layer3Controller(hz=50)
    perception = MockPerception()
    
    # 場景：從 (0,0,0) 前往 (10,10,-10)
    # 路徑中間 (5,5,-10) 有一個障礙物
    target = Vector3(10, 10, -10)
    controller.target = target
    
    print("="*60)
    print("  Layer 3 Low-Level Controller - 避障展示 Demo")
    print("="*60)
    print(f"任務：前往目標點 {target}")
    print(f"環境：在 (5, 5, -10) 存在一棵樹（障礙物）")
    print("-"*60)
    
    time.sleep(2)
    
    steps = 0
    max_steps = 1000
    
    while steps < max_steps:
        # 1. 感知環境
        obs = perception.get_nearby_obstacles(controller.pos.x, controller.pos.y, controller.pos.z, radius=5.0)
        controller.obstacles = obs
        
        # 2. 控制步進
        pos, vel = controller.step()
        steps += 1
        
        # 3. 顯示狀態 (每 10 步顯示一次)
        if steps % 10 == 0:
            dist_to_target = pos.dist(target)
            closest_obs_dist = 999.0
            if obs:
                closest_obs_dist = min([pos.dist(Vector3(o['x'], o['y'], o['z'])) for o in obs])
            
            # 使用簡單的進度條顯示
            progress = int((1 - dist_to_target / 14.0) * 20)
            bar = "█" * max(0, progress) + "░" * max(0, 20 - progress)
            
            print(f"Step {steps:03} | 位置: ({pos.x:5.2f}, {pos.y:5.2f}, {pos.z:5.2f}) | "
                  f"速度: {vel.norm():4.2f}m/s | "
                  f"距離障礙: {closest_obs_dist:4.2f}m | {bar}")
            
        if pos.dist(target) < 0.3:
            print("\n✅ 成功抵達目標點，任務完成！")
            break
            
        time.sleep(0.02) # 50Hz 模擬

if __name__ == "__main__":
    run_demo_scene()
