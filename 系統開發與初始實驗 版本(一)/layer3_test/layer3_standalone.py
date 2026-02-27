"""
layer3_standalone.py - Layer 3 Low-Level Flight Controller（獨立測試版）
========================================================================
根據論文「基於仿生神經分工架構之生成式 AI 自主無人機研究」第 4.3 節設計。

核心功能：
  1. 人工勢場法 (APF) - 實時計算引力（目標點）與斥力（障礙物）
  2. PID 控制器        - 穩定速度輸出，防止姿態劇烈震盪
  3. 50Hz 高頻控制迴圈 - 模擬模擬器 30-50Hz 的即時控制需求
  4. 向量場整合        - 合併 Layer 2 的航點引導與本地避障斥力

本模組為獨立開發版本，內建物理模擬，無需 AirSim 環境即可測試。
"""

import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def to_np(self):
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_np(cls, arr):
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))

    def dist(self, other: 'Vector3') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def norm(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        n = self.norm()
        if n < 1e-6: return Vector3(0,0,0)
        return Vector3(self.x/n, self.y/n, self.z/n)

class PIDController:
    """PID 控制器，支撐論文提到的 PID 增益自動調校基礎"""
    def __init__(self, kp: float, ki: float, kd: float, limit: float = 5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        
        self.prev_error = 0.0
        self.integral = 0.0
        
    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        # 簡單的積分飽和防護
        self.integral = max(min(self.integral, self.limit), -self.limit)
        
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = error
        # 限制輸出速度 (論文指定速度上限 5.0 m/s)
        return max(min(output, self.limit), -self.limit)

class APFEngine:
    """
    人工勢場法引擎 (Artificial Potential Field)
    計算引力 (Attractive) 與斥力 (Repulsive) 的合力
    """
    def __init__(self, k_att: float = 1.0, k_rep: float = 50.0, rho_0: float = 2.5):
        self.k_att = k_att      # 引力增益
        self.k_rep = k_rep      # 斥力增益
        self.rho_0 = rho_0      # 斥力影響範圍 (rho_0 = 2m-2.5m)

    def calculate_force(self, current_pos: Vector3, target_pos: Vector3, obstacles: List[Dict]) -> Vector3:
        # 1. 計算引力 (線性引力)
        # F_att = k_att * (target - current)
        force_att = (target_pos - current_pos) * self.k_att
        
        # 2. 計算斥力
        force_rep = Vector3(0, 0, 0)
        for obs in obstacles:
            obs_pos = Vector3(obs['x'], obs['y'], obs['z'])
            dist = current_pos.dist(obs_pos)
            
            if dist < self.rho_0:
                # 論文基礎公式：F_rep = k_rep * (1/dist - 1/rho_0) * (1/dist^2) * unit_vector
                # 為了數值穩定，避免 dist=0
                safe_dist = max(dist, 0.1)
                rep_mag = self.k_rep * (1.0/safe_dist - 1.0/self.rho_0) * (1.0/(safe_dist**2))
                
                direction_vec = current_pos - obs_pos
                if direction_vec.norm() < 1e-4:
                    direction_vec = Vector3(0.1, 0.1, 0) # 避免除以零
                    
                direction = direction_vec.normalize()
                
                # [改進] 增加微小的正交分力 (Orthogonal Force) 以打破局部極小值 (Local Minima) 重線死鎖
                # 旋轉 90 度得到 xy 平面上的法向量
                ortho_dir = Vector3(-direction.y, direction.x, 0).normalize()
                if ortho_dir.norm() < 1e-4:
                    ortho_dir = Vector3(1, 0, 0)
                
                # 合併主斥力與 20% 的側向偏移力
                combined_rep_dir = direction + (ortho_dir * 0.2)
                
                force_rep = force_rep + (combined_rep_dir * rep_mag)
        
        total_force = force_att + force_rep
        
        # [NEW] U-Shape Local Minima Escape (Wall-Following)
        if force_rep.norm() > 0.5 and force_att.norm() > 0.1:
            att_dir = force_att.normalize()
            rep_dir = force_rep.normalize()
            # 判斷引力與總斥力是否幾乎完全反向 (夾角接近 180度, 點積 < -0.9)
            dot_prod = att_dir.x * rep_dir.x + att_dir.y * rep_dir.y
            
            # 若陷入深度對抗死鎖 (總力極小) 且強力互斥
            if dot_prod < -0.9 and total_force.norm() < force_att.norm() * 0.4:
                # 啟動強制沿牆逃脫 (Orthogonal Wall-Following)
                # 利用旋轉 90 度產生切線向量
                escape_dir = Vector3(-rep_dir.y, rep_dir.x, 0).normalize()
                # 覆蓋原力，給予強大的側向切線力道以脫離 U 型區
                total_force = escape_dir * force_rep.norm() * 1.5
                
        return total_force

class Layer3Controller:
    """Layer 3 主控制器（50Hz 頻率）"""
    def __init__(self, hz: int = 50):
        self.hz = hz
        self.dt = 1.0 / hz
        
        self.apf = APFEngine()
        self.pid_x = PIDController(kp=1.2, ki=0.1, kd=0.05)
        self.pid_y = PIDController(kp=1.2, ki=0.1, kd=0.05)
        self.pid_z = PIDController(kp=1.5, ki=0.1, kd=0.1)
        
        # 狀態
        self.pos = Vector3(0, 0, 0)
        self.vel = Vector3(0, 0, 0)
        self.target = Vector3(0, 0, 0)
        self.obstacles = []
        
        self.is_running = False
        
    def step(self):
        """執行單步控制計算"""
        # 1. 獲取 APF 合力向量
        force = self.apf.calculate_force(self.pos, self.target, self.obstacles)
        
        # 2. PID 轉換為速度指令
        vx = self.pid_x.update(force.x, self.dt)
        vy = self.pid_y.update(force.y, self.dt)
        vz = self.pid_z.update(force.z, self.dt)
        
        self.vel = Vector3(vx, vy, vz)
        
        # 3. 物理模擬（SITL 模擬位移）
        self.pos.x += self.vel.x * self.dt
        self.pos.y += self.vel.y * self.dt
        self.pos.z += self.vel.z * self.dt
        
        return self.pos, self.vel

    def run_simulation(self, target: Vector3, obstacles: List[Dict], steps: int = 100):
        self.target = target
        self.obstacles = obstacles
        self.is_running = True
        
        path = []
        for _ in range(steps):
            p, v = self.step()
            path.append((p.x, p.y, p.z))
            if p.dist(target) < 0.2:
                break
            time.sleep(self.dt / 10) # 加速模擬
            
        return path

if __name__ == "__main__":
    controller = Layer3Controller()
    target = Vector3(10, 10, -10)
    # 放置一個障礙物在路徑中間 (5, 5, -10)
    obs = [{"x": 5.0, "y": 5.0, "z": -10.0}]
    
    print(f"🚀 開始 Layer 3 避障模擬...")
    print(f"起點: {controller.pos} | 目標: {target}")
    
    path = controller.run_simulation(target, obs, steps=500)
    
    print(f"🏁 模擬完成，總步數: {len(path)}")
    print(f"最終位置: {controller.pos}")
    dist_to_obs = controller.pos.dist(Vector3(5,5,-10))
    print(f"與最近障礙物距離: {dist_to_obs:.2f}m")
