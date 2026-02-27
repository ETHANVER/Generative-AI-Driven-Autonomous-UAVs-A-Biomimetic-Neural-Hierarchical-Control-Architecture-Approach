"""
test_layer3.py - Layer 3 自動化測試套件
========================================
驗證 APF 避障邏輯、PID 輸出限制以及 50Hz 控制穩定性。
"""

import unittest
import numpy as np
from layer3_standalone import Layer3Controller, Vector3, PIDController, APFEngine

class TestLayer3(unittest.TestCase):
    def setUp(self):
        self.controller = Layer3Controller(hz=50)

    def test_pid_limit(self):
        """TC01: 驗證速度輸出是否被限制在 5m/s (論文 4.3 節)"""
        pid = PIDController(kp=100.0, ki=0, kd=0, limit=5.0) 
        output = pid.update(error=1000.0, dt=0.02) # 極大誤差
        self.assertEqual(output, 5.0)
        output_neg = pid.update(error=-1000.0, dt=0.02)
        self.assertEqual(output_neg, -5.0)

    def test_apf_attraction(self):
        """TC02: 驗證引力方向是否朝向目標"""
        engine = APFEngine(k_att=1.0, k_rep=0.0) # 關閉斥力
        curr = Vector3(0, 0, 0)
        target = Vector3(10, 0, 0)
        force = engine.calculate_force(curr, target, [])
        self.assertGreater(force.x, 0)
        self.assertEqual(force.y, 0)
        self.assertEqual(force.z, 0)

    def test_apf_repulsion(self):
        """TC03: 驗證斥力是否正確觸發 (rho_0 = 2.5m)"""
        engine = APFEngine(k_att=0.0, k_rep=100.0, rho_0=2.5) # 僅計算斥力
        curr = Vector3(1.0, 0, 0)
        obs = [{"x": 0.0, "y": 0.0, "z": 0.0}] # 障礙物在 1m 外
        
        force = engine.calculate_force(curr, Vector3(10,0,0), obs)
        # 斥力方向應背對障礙物 (沿 +X 軸)
        self.assertGreater(force.x, 0)
        
        # 測試範圍外不觸發
        curr_far = Vector3(5.0, 0, 0)
        force_far = engine.calculate_force(curr_far, Vector3(10,0,0), obs)
        self.assertEqual(force_far.x, 0)

    def test_full_navigation(self):
        """TC04: 驗證從 (0,0,0) 到 (5,5,0) 的完整導航"""
        target = Vector3(5, 5, 0)
        path = self.controller.run_simulation(target, [], steps=500)
        
        final_dist = self.controller.pos.dist(target)
        self.assertLess(final_dist, 0.3, "未抵達目標點範圍")

    def test_obstacle_avoidance(self):
        """TC05: 驗證避障偏移 (打破重線局部極小值)"""
        # 直線前進 (0,0,0) -> (10,0,0)，在 (5,0,0) 放障礙物
        self.controller.pos = Vector3(0, 0, 0)
        target = Vector3(10, 0, 0)
        obs = [{"x": 5.0, "y": 0.0, "z": 0.0}]
        
        self.controller.target = target
        self.controller.obstacles = obs
        
        max_y_offset = 0.0
        
        # 模擬 500 步
        for _ in range(500):
            p, v = self.controller.step()
            if abs(p.y) > max_y_offset:
                max_y_offset = abs(p.y)
                
            # 提早抵達目標點
            if p.dist(target) < 0.5:
                break
                
        # 1. 驗證是否由於正交力成功產生了橫向位移 (大於 0.1)
        self.assertGreater(max_y_offset, 0.1, "避障未產生橫向偏移 (陷入局部極小值)")
        
        # 2. 驗證最終是否能繞過障礙物抵達目標
        final_dist = self.controller.pos.dist(target)
        self.assertLess(final_dist, 1.0, "未能成功繞開障礙物抵達目標")

if __name__ == "__main__":
    unittest.main()
