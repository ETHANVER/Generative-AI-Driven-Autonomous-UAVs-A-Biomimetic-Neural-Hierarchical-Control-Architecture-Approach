"""
mock_perception.py - 模擬感知系統
=================================
模擬 AirSim 的深度攝像頭或物件辨識系統，
為 Layer 3 提供即時的障礙物座標。
"""

import random
from typing import List, Dict

class MockPerception:
    def __init__(self):
        # 預設障礙物清單 (x, y, z)
        self.static_obstacles = [
            {"x": 5.0, "y": 5.0, "z": -10.0, "label": "tree"},
            {"x": 8.0, "y": 2.0, "z": -12.0, "label": "wall"},
            {"x": 15.0, "y": 15.0, "z": -5.0, "label": "no_fly_zone_edge"},
        ]

    def get_nearby_obstacles(self, current_pos_x: float, current_pos_y: float, current_pos_z: float, radius: float = 5.0) -> List[Dict]:
        """
        模擬攝像頭視野，僅回傳半徑內的障礙物。
        """
        nearby = []
        for obs in self.static_obstacles:
            dist = ((obs['x'] - current_pos_x)**2 + 
                    (obs['y'] - current_pos_y)**2 + 
                    (obs['z'] - current_pos_z)**2)**0.5
            if dist <= radius:
                # 加入一些感測器雜訊 (±1cm)
                noise_obs = obs.copy()
                noise_obs['x'] += random.uniform(-0.01, 0.01)
                noise_obs['y'] += random.uniform(-0.01, 0.01)
                nearby.append(noise_obs)
        return nearby

    def add_dynamic_obstacle(self, x, y, z, label="dynamic"):
        self.static_obstacles.append({"x": x, "y": y, "z": z, "label": label})
