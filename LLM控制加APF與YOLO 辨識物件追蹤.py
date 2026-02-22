import airsim
import cv2
import numpy as np
import math
import time
import torch
import json
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox
from collections import deque, Counter
from torchvision import models, transforms
from ultralytics import YOLOWorld
from groq import Groq

# ==========================================
# 0. 系統參數
# ==========================================
GROQ_API_KEY = ""

# 飛行參數
FLIGHT_ALTITUDE = -3.0   
MAX_SPEED = 2.0          
YOLO_CONFIDENCE = 0.4    

# 尋路參數
SEEK_TRIGGER_DIST = 10.0
GAP_SENSITIVITY = 2.0    
EMERGENCY_DIST = 2.0     

# 安全參數
SAFE_YAW_THRESHOLD = 15.0  
NOSE_FORWARD_SPEED = 0.5   

# 視覺避障參數
VISUAL_WARN_THRESHOLD = 0.10
VISUAL_STOP_THRESHOLD = 0.20

# 記憶參數
GRID_SIZE = 0.5          
MAP_SIZE_PIXELS = 800    
METERS_PER_PIXEL = 0.1   

# YOLO 目標確認參數
YOLO_CONFIRMATION_FRAMES = 3  
YOLO_TRACKING_DISTANCE = 2.0  

# 全域變數
GLOBAL_STATE = {
    "perception_text": "",      
    "current_action": "MOVE_FORWARD",
    "current_action_params": {},  
    "current_reasoning": "Initializing system...",
    "current_observation": "Waiting for data...",
    "new_data_available": False,
    "last_llm_time": 0,
    "visual_debug_info": {}
}

# ==========================================
# 動作執行器 (修正座標系 + 加入 APPROACH 支援)
# ==========================================
class ActionExecutor:
    """負責執行具體的動作指令"""
    def __init__(self, client):
        self.client = client
        self.current_action = None
        self.action_start_time = None
        self.action_start_pos = None
        self.action_start_yaw = None
        self.is_executing = False
        
    def start_action(self, action_type, params, current_pos, current_yaw):
        """開始執行一個新動作"""
        self.current_action = action_type
        self.action_params = params
        self.action_start_time = time.time()
        self.action_start_pos = current_pos
        self.action_start_yaw = current_yaw
        self.is_executing = True
        
        print(f"🎯 開始執行: {action_type} with params: {params}")
    
    def check_completion(self, current_pos, current_yaw):
        """檢查動作是否完成"""
        if not self.is_executing:
            return True
            
        # [修正] 整合 MOVE_FORWARD 與 APPROACH 的完成判斷
        if self.current_action in ["MOVE_FORWARD", "APPROACH"]:
            raw_dist = self.action_params.get("distance", 10.0)
            
            # 對於 APPROACH，自動扣除 3公尺 的安全煞車距離
            # 如果目標在 22.7m，我們只飛 19.7m，避免直接撞上
            target_dist = raw_dist
            if self.current_action == "APPROACH":
                target_dist = max(1.0, raw_dist - 3.0)

            traveled = math.sqrt(
                (current_pos.x_val - self.action_start_pos.x_val)**2 +
                (current_pos.y_val - self.action_start_pos.y_val)**2
            )
            if traveled >= target_dist * 0.9:  # 90% 完成度
                self.is_executing = False
                print(f"✅ 完成 {self.current_action} (移動 {traveled:.1f}m)")
                return True
                
        elif self.current_action in ["ROTATE_LEFT", "ROTATE_RIGHT"]:
            target_angle = self.action_params.get("angle", 45.0)
            yaw_diff = abs(math.degrees(current_yaw - self.action_start_yaw))
            if yaw_diff >= target_angle * 0.9:
                self.is_executing = False
                print(f"✅ 完成旋轉 {yaw_diff:.1f}度")
                return True
                
        elif self.current_action in ["STRAFE_LEFT", "STRAFE_RIGHT"]:
            distance = self.action_params.get("distance", 5.0)
            traveled = math.sqrt(
                (current_pos.x_val - self.action_start_pos.x_val)**2 +
                (current_pos.y_val - self.action_start_pos.y_val)**2
            )
            if traveled >= distance * 0.9:
                self.is_executing = False
                print(f"✅ 完成平移 {traveled:.1f}m")
                return True
        
        # 超時保護 (20秒)
        if time.time() - self.action_start_time > 20.0:
            self.is_executing = False
            print("⚠️ 動作超時，強制完成")
            return True
            
        return False
    
    def get_velocity_command(self, current_yaw):
        """根據當前動作生成速度指令"""
        if not self.is_executing:
            return 0, 0, 0
        
        # [修正] 讓 APPROACH 也能產生前進速度
        if self.current_action in ["MOVE_FORWARD", "APPROACH"]:
            speed = self.action_params.get("speed", 1.5)
            vx = speed * math.cos(current_yaw)
            vy = speed * math.sin(current_yaw)
            return vx, vy, 0
            
        elif self.current_action == "ROTATE_LEFT":
            return 0, 0, -30.0
            
        elif self.current_action == "ROTATE_RIGHT":
            return 0, 0, 30.0
            
        elif self.current_action == "STRAFE_LEFT":
            speed = self.action_params.get("speed", 1.0)
            # 向左平移 (-90度)
            vx = speed * math.cos(current_yaw - math.pi/2)
            vy = speed * math.sin(current_yaw - math.pi/2)
            return vx, vy, 0
            
        elif self.current_action == "STRAFE_RIGHT":
            speed = self.action_params.get("speed", 1.0)
            # 向右平移 (+90度)
            vx = speed * math.cos(current_yaw + math.pi/2)
            vy = speed * math.sin(current_yaw + math.pi/2)
            return vx, vy, 0
            
        return 0, 0, 0

# ==========================================
# YOLO 目標追蹤器
# ==========================================
class YOLOTargetTracker:
    def __init__(self, confirmation_frames=3, max_distance=2.0):
        self.confirmation_frames = confirmation_frames
        self.max_distance = max_distance
        self.pending_targets = {} 
        self.confirmed_targets = {} 
        
    def update(self, detections, drone_pos, drone_yaw):
        current_frame_labels = set()
        
        for det in detections:
            label = det['label']
            world_pos = det['world_pos']
            current_frame_labels.add(label)
            
            if label not in self.pending_targets:
                self.pending_targets[label] = []
            
            matched = False
            for pending in self.pending_targets[label]:
                dist = math.sqrt(
                    (world_pos[0] - pending['pos'][0])**2 +
                    (world_pos[1] - pending['pos'][1])**2
                )
                if dist < self.max_distance:
                    pending['frame_count'] += 1
                    pending['pos'] = world_pos 
                    matched = True
                    
                    if pending['frame_count'] >= self.confirmation_frames:
                        self._confirm_target(label, world_pos)
                        self.pending_targets[label].remove(pending)
                    break
            
            if not matched:
                self.pending_targets[label].append({
                    'pos': world_pos,
                    'frame_count': 1
                })
        
        for label in list(self.pending_targets.keys()):
            self.pending_targets[label] = [
                p for p in self.pending_targets[label] 
                if p['frame_count'] > 0
            ]
            for p in self.pending_targets[label]:
                if label not in current_frame_labels:
                    p['frame_count'] = max(0, p['frame_count'] - 1)
    
    def _confirm_target(self, label, pos):
        if label not in self.confirmed_targets:
            self.confirmed_targets[label] = []
        
        is_duplicate = False
        for confirmed_pos, _ in self.confirmed_targets[label]:
            dist = math.sqrt(
                (pos[0] - confirmed_pos[0])**2 +
                (pos[1] - confirmed_pos[1])**2
            )
            if dist < self.max_distance * 1.5:
                is_duplicate = True
                break
        
        if not is_duplicate:
            self.confirmed_targets[label].append((pos, time.time()))
            print(f"✨ 確認新目標: {label.upper()} at ({pos[0]:.1f}, {pos[1]:.1f})")
    
    def get_confirmed_targets(self):
        return self.confirmed_targets

# ==========================================
# UI: 任務設定 (CoT) - [策略指令修正]
# ==========================================
class MissionConfigDialog:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.target_classes = "car"
        
        self.mission_prompt = (
            "You are an AI Drone Commander.\n"
            "Goal: Search for 'car' objects while avoiding obstacles.\n"
            # ----- [策略指令修正] -----
            "Strategy: Prioritize MOVE_FORWARD if Front > 5m. "
            "If Front < 4m (Blocked):\n"
            "1. If this is a new obstacle, use STRAFE_LEFT or STRAFE_RIGHT to bypass.\n"
            "2. If you just STRAFED and are STILL blocked, it means the obstacle is large (Wall). Do NOT strafe again. Use ROTATE instead.\n"
            "3. Only use ROTATE (Turn) if the path is completely blocked (Dead end or Wall).\n"
            # ------------------------
            "\n"
            "Process:\n"
            "1. ANALYZE: Check spatial status (Front/Left/Right), Exploration Status, and detected objects.\n"
            "2. REASON: If blocked, find open space. If target 'car' seen, approach it. If a sector is unexplored, prioritize it.\n"
            "3. DECIDE: Pick one action with specific parameters.\n"
            "\n"
            "Available Actions:\n"
            "- MOVE_FORWARD: {\"distance\": 10, \"speed\": 1.5}\n"
            "- ROTATE_LEFT: {\"angle\": 45}\n"
            "- ROTATE_RIGHT: {\"angle\": 45}\n"
            "- STRAFE_LEFT: {\"distance\": 2, \"speed\": 1.0}\n"
            "- STRAFE_RIGHT: {\"distance\": 2, \"speed\": 1.0}\n"
            "- APPROACH: {\"target\": \"car\", \"distance\": 5}\n"
            "- STOP: {}\n"
            "\n"
            "Output JSON format:\n"
            "{\n"
            "  \"observation\": \"Front is blocked (<4m), but I see Left side is open.\",\n"
            "  \"reasoning\": \"To bypass the obstacle without turning, I will strafe left.\",\n"
            "  \"action\": \"STRAFE_LEFT\",\n"
            "  \"params\": {\"distance\": 5}\n"
            "}"
        )

    def get_input(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Mission Configuration")
        dialog.geometry("600x550")
        
        tk.Label(dialog, text="偵察目標類別 (逗號分隔):", font=("Arial", 12, "bold")).pack(pady=5)
        entry_classes = tk.Entry(dialog, width=60, font=("Arial", 10))
        entry_classes.insert(0, self.target_classes)
        entry_classes.pack(pady=5)
        
        tk.Label(dialog, text="任務指令 (System Prompt with CoT):", font=("Arial", 12, "bold")).pack(pady=5)
        text_prompt = tk.Text(dialog, height=20, width=70, font=("Arial", 10))
        text_prompt.insert("1.0", self.mission_prompt)
        text_prompt.pack(pady=5)
        
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        result = {}
        def on_submit():
            result['classes'] = [x.strip() for x in entry_classes.get().split(',')]
            result['prompt'] = text_prompt.get("1.0", tk.END).strip()
            dialog.destroy()
            self.root.destroy()

        tk.Button(btn_frame, text="Start Mission", command=on_submit, bg="green", fg="white", width=20).pack()
        self.root.wait_window(dialog)
        
        if not result:
            return [x.strip() for x in self.target_classes.split(',')], self.mission_prompt
        return result['classes'], result['prompt']

# ==========================================
# 路徑記憶 (Path Memory)
# ==========================================
class PathMemory:
    def __init__(self, grid_size=0.5, time_decay=300.0):
        self.grid_size = grid_size
        self.time_decay = time_decay
        self.visited_cells = {} 
        self.position_history = deque(maxlen=300) 
        self.repetition_count = {}
        
    def update_position(self, x, y, current_time):
        gx = int(x / self.grid_size)
        gy = int(y / self.grid_size)
        cell = (gx, gy)
        
        expired_cells = [k for k, t in self.visited_cells.items() 
                        if (current_time - t) > self.time_decay]
        for cell_key in expired_cells:
            del self.visited_cells[cell_key]
            if cell_key in self.repetition_count:
                del self.repetition_count[cell_key]
        
        self.repetition_count[cell] = self.repetition_count.get(cell, 0) + 1
        self.visited_cells[cell] = current_time
        self.position_history.append((x, y))
    
    def get_repulsion_force(self, x, y, current_time, radius=4):
        gx = int(x / self.grid_size)
        gy = int(y / self.grid_size)
        fx, fy = 0.0, 0.0
        SAFE_DIST = 2.5
        DANGER_DIST = 1.0
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_cell = (gx + dx, gy + dy)
                if check_cell not in self.visited_cells: continue
                
                dist_sq = (dx**2 + dy**2) * (self.grid_size**2)
                dist = math.sqrt(dist_sq)
                if dist <= 0.1: dist = 0.1
                if dist > SAFE_DIST: continue
                
                rep_factor = min(3.0, self.repetition_count.get(check_cell, 1) * 0.5)
                
                if dist < DANGER_DIST:
                    strength = rep_factor * (2.0 / dist) ** 2
                else:
                    strength = rep_factor * 0.5 * (SAFE_DIST - dist) / (SAFE_DIST - DANGER_DIST)
                
                fx += -(dx / dist) * strength
                fy += -(dy / dist) * strength
        return fx, fy

# ==========================================
# 覆蓋地圖 (Coverage Map)
# ==========================================
class CoverageMap:
    def __init__(self, size_meters=400, cell_size=2.0):
        self.size = size_meters
        self.cell_size = cell_size
        self.grid_dim = int(size_meters / cell_size)
        self.center_idx = self.grid_dim // 2
        self.grid = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)
        self.last_grid_pos = None

    def world_to_grid(self, x, y):
        gx = int(x / self.cell_size) + self.center_idx
        gy = int(y / self.cell_size) + self.center_idx
        return gx, gy

    def update(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        if 0 <= gx < self.grid_dim and 0 <= gy < self.grid_dim:
            if (gx, gy) != self.last_grid_pos:
                try:
                    self.grid[gy-1:gy+2, gx-1:gx+2] += 0.5 
                    self.grid[gy, gx] += 1.0 
                except: pass
                self.last_grid_pos = (gx, gy)

    def get_repulsive_force(self, current_x, current_y):
        gx, gy = self.world_to_grid(current_x, current_y)
        window = 5 
        y_min = max(0, gy-window); y_max = min(self.grid_dim, gy+window)
        x_min = max(0, gx-window); x_max = min(self.grid_dim, gx+window)
        local = self.grid[y_min:y_max, x_min:x_max]
        if local.size == 0: return 0.0, 0.0
        grad_y, grad_x = np.gradient(local)
        return -np.mean(grad_x) * 2.0, -np.mean(grad_y) * 2.0

    def get_exploration_stats(self):
        c = self.center_idx
        quad_area = (self.grid_dim // 2) ** 2
        if quad_area == 0: quad_area = 1
        
        ne_grid = self.grid[c:, c:]
        nw_grid = self.grid[c:, :c]
        se_grid = self.grid[:c, c:]
        sw_grid = self.grid[:c, :c]
        
        stats = {
            "North-East": (np.count_nonzero(ne_grid) / quad_area) * 100,
            "North-West": (np.count_nonzero(nw_grid) / quad_area) * 100,
            "South-East": (np.count_nonzero(se_grid) / quad_area) * 100,
            "South-West": (np.count_nonzero(sw_grid) / quad_area) * 100
        }
        
        return ", ".join([f"{k}: {v:.1f}%" for k, v in stats.items()])

# ==========================================
# 規劃器
# ==========================================
class GapSeekingPlanner:
    def __init__(self):
        self.smooth_factor = 0.08 
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_yaw_rate = 0.0
    
    def calculate_global_velocity(self, target_speed, heading_bias, gaps, drone_yaw, 
                                  curr_x, curr_y, coverage_map, path_repulsion, yolo_detections, img_width):
        
        dist_L, dist_C, dist_R = gaps
        if dist_C == 0: dist_C = 50.0 

        # 1. 基礎吸引力
        target_angle_rad = math.radians(heading_bias)
        attr_x = math.cos(target_angle_rad) * target_speed 
        attr_y = math.sin(target_angle_rad) * target_speed 

        repulse_x, repulse_y = 0.0, 0.0
        override_yaw_rate = None
        
        vis_debug = {"active": False, "target_box": None, "msg": ""}

        # 視覺避障
        vis_repulse_y = 0.0
        center_x = img_width // 2
        trees = [d for d in yolo_detections if d['label'] == 'tree']
        
        if trees:
            largest_tree = max(trees, key=lambda x: (x['roi'][2]-x['roi'][0]) * (x['roi'][3]-x['roi'][1]))
            box = largest_tree['roi']
            box_w = box[2] - box[0]
            box_cx = (box[0] + box[2]) / 2
            
            width_ratio = box_w / img_width
            offset = (box_cx - center_x) / (img_width / 2) 
            
            if abs(offset) < 0.6:
                if width_ratio > VISUAL_WARN_THRESHOLD: 
                    vis_debug["active"] = True
                    vis_debug["target_box"] = box
                    
                    push_dir = 1.0 if offset <= 0 else -1.0
                    push_strength = 5.0 * width_ratio 
                    vis_repulse_y += push_strength * push_dir
                    
                    if width_ratio > VISUAL_STOP_THRESHOLD:
                        vis_debug["msg"] = "EMERGENCY STOP & SLIDE"
                        attr_x = 0.0 
                        repulse_x = -1.0 
                    else:
                        vis_debug["msg"] = "DODGING TREE"
                        repulse_x -= 1.0 

        repulse_y += vis_repulse_y

        # 2. 局部避障
        if dist_C < SEEK_TRIGGER_DIST:
            repulse_x -= (SEEK_TRIGGER_DIST - dist_C) * 0.7 
            
            if abs(vis_repulse_y) < 0.5:
                gap_diff = dist_L - dist_R
                repulse_y -= gap_diff * 0.2
                if gap_diff > 3.0: override_yaw_rate = -15.0
                elif gap_diff < -3.0: override_yaw_rate = 15.0
        
        # 3. 主動居中
        if dist_L < 4.0 and dist_R < 4.0:
            diff = dist_R - dist_L
            if abs(diff) > 0.5: repulse_y += diff * 0.3

        # 4. Coverage 斥力
        current_speed = math.sqrt(self.last_vx**2 + self.last_vy**2)
        if current_speed < 0.5: 
            cov_rep_x, cov_rep_y = coverage_map.get_repulsive_force(curr_x, curr_y)
            body_cov_x = cov_rep_x * math.cos(-drone_yaw) - cov_rep_y * math.sin(-drone_yaw)
            body_cov_y = cov_rep_x * math.sin(-drone_yaw) + cov_rep_y * math.cos(-drone_yaw)
            repulse_x += body_cov_x * 0.2
            repulse_y += body_cov_y * 0.2

        # 5. Path Memory 斥力
        path_fx, path_fy = path_repulsion
        if abs(path_fx) > 0 or abs(path_fy) > 0:
             body_path_x = path_fx * math.cos(-drone_yaw) - path_fy * math.sin(-drone_yaw)
             body_path_y = path_fx * math.sin(-drone_yaw) + path_fy * math.cos(-drone_yaw)
             repulse_x += body_path_x * 0.8 
             repulse_y += body_path_y * 0.8

        # 6. 合成
        body_vx = attr_x + repulse_x
        body_vy = attr_y + repulse_y
        if body_vx < -0.5: body_vx = -0.5 
        speed = math.sqrt(body_vx**2 + body_vy**2)
        if speed > MAX_SPEED:
            scale = MAX_SPEED / speed
            body_vx *= scale; body_vy *= scale

        global_vx = body_vx * math.cos(drone_yaw) - body_vy * math.sin(drone_yaw)
        global_vy = body_vx * math.sin(drone_yaw) + body_vy * math.cos(drone_yaw)

        # 機頭鎖定與安全轉向
        if override_yaw_rate is not None:
            target_yaw_rate = override_yaw_rate
        else:
            target_yaw_rate = heading_bias * 1.5

        # Nose Forward
        global_speed = math.sqrt(global_vx**2 + global_vy**2)
        if global_speed > NOSE_FORWARD_SPEED:
            travel_angle = math.atan2(global_vy, global_vx)
            angle_diff = travel_angle - drone_yaw
            while angle_diff > math.pi: angle_diff -= 2*math.pi
            while angle_diff < -math.pi: angle_diff += 2*math.pi
            align_yaw_rate = math.degrees(angle_diff) * 1.5
            target_yaw_rate = align_yaw_rate

        target_yaw_rate = max(-40.0, min(40.0, target_yaw_rate))

        # Safety Turn Brake
        if abs(target_yaw_rate) > SAFE_YAW_THRESHOLD:
            brake_factor = max(0.0, 1.0 - (abs(target_yaw_rate) - SAFE_YAW_THRESHOLD) / 10.0)
            global_vx *= brake_factor
            global_vy *= brake_factor

        final_vx = self.smooth_factor * global_vx + (1 - self.smooth_factor) * self.last_vx
        final_vy = self.smooth_factor * global_vy + (1 - self.smooth_factor) * self.last_vy
        final_yaw_rate = self.smooth_factor * target_yaw_rate + (1 - self.smooth_factor) * self.last_yaw_rate
        
        self.last_vx = final_vx; self.last_vy = final_vy; self.last_yaw_rate = final_yaw_rate
        
        return final_vx, final_vy, final_yaw_rate, vis_debug

# ==========================================
# LLM Brain
# ==========================================
class AsyncLLMBrain(threading.Thread):
    def __init__(self, api_key, system_prompt):
        super().__init__()
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
        self.daemon = True 
        self.system_prompt = system_prompt

    def run(self):
        print("🧠 [Background] LLM Brain Started.")
        while True:
            if GLOBAL_STATE["new_data_available"] and (time.time() - GLOBAL_STATE["last_llm_time"] > 2.0):
                prompt = GLOBAL_STATE["perception_text"]
                GLOBAL_STATE["new_data_available"] = False
                try:
                    chat = self.client.chat.completions.create(
                        messages=[{"role": "system", "content": self.system_prompt},
                                  {"role": "user", "content": prompt}],
                        model=self.model, temperature=0.2, max_tokens=200,
                        response_format={"type": "json_object"},
                    )
                    res = json.loads(chat.choices[0].message.content)
                    GLOBAL_STATE["current_action"] = res.get("action", "MOVE_FORWARD")
                    GLOBAL_STATE["current_action_params"] = res.get("params", {}) 
                    GLOBAL_STATE["current_reasoning"] = res.get("reasoning", "No reasoning.")
                    GLOBAL_STATE["current_observation"] = res.get("observation", "")
                    GLOBAL_STATE["last_llm_time"] = time.time()
                    
                    print(f"\n👀 [Obs]: {GLOBAL_STATE['current_observation']}")
                    print(f"🧠 [Think]: {GLOBAL_STATE['current_reasoning']}")
                    print(f"⚡ [Cmd]: {GLOBAL_STATE['current_action']} {GLOBAL_STATE['current_action_params']}")
                except Exception as e: 
                    print(f"🔨 LLM Error: {e}")
            time.sleep(0.1)

# ==========================================
# 視覺與記憶
# ==========================================
class FastSceneClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.model = models.mobilenet_v3_large(weights=weights).to(self.device).eval()
        self.categories = weights.meta["categories"]
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize(256),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.history = deque(maxlen=10)

    def predict(self, cv2_img):
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(rgb_img).unsqueeze(0).to(self.device)
        with torch.no_grad(): output = self.model(input_tensor)
        top3_labels = [self.categories[i].lower() for i in torch.topk(torch.nn.functional.softmax(output[0], dim=0), 3)[1]]
        
        scene_hint = "Unknown"
        urban_keywords = ['building', 'house', 'street', 'road']
        nature_keywords = ['forest', 'mountain', 'tree', 'lakeside']
        
        if any(u in label for label in top3_labels for u in urban_keywords): scene_hint = "Urban Area"
        elif any(n in label for label in top3_labels for n in nature_keywords): scene_hint = "Nature"
        self.history.append(scene_hint)
        return Counter(self.history).most_common(1)[0][0]

class RobustMemorySystem:
    COLORS = {"tree": (0, 255, 0), "car": (0, 165, 255), "building": (0, 0, 255), "person": (255, 0, 255), "road": (50, 50, 50), "unknown": (150, 150, 150)}
    
    def __init__(self):
        self.decay_rate = 1.0
        self.add_rate = 30.0
        self.max_conf = 255.0
        self.grid_map = {} 
    
    def update(self, drone_pos, drone_yaw, relative_points):
        keys_to_del = [k for k, v in self.grid_map.items() if (v['conf'] - self.decay_rate) <= 0]
        for k in keys_to_del: del self.grid_map[k]
        for k in self.grid_map: self.grid_map[k]['conf'] -= self.decay_rate
        
        dx, dy = drone_pos.x_val, drone_pos.y_val
        cos_yaw, sin_yaw = math.cos(drone_yaw), math.sin(drone_yaw)
        for (dist, lat, label) in relative_points:
            wx = dx + (dist * cos_yaw - lat * sin_yaw)
            wy = dy + (dist * sin_yaw + lat * cos_yaw)
            gx, gy = int(wx / GRID_SIZE), int(wy / GRID_SIZE)
            if (gx, gy) not in self.grid_map: self.grid_map[(gx, gy)] = {'conf': 0.0, 'votes': {}}
            cell = self.grid_map[(gx, gy)]
            cell['conf'] = min(self.max_conf, cell['conf'] + self.add_rate)
            cell['votes'][label] = cell['votes'].get(label, 0) + 1
    
    def generate_llm_prompt(self, drone_pos, drone_yaw, scene_context, gaps, last_action, exploration_status, confirmed_targets):
        dx, dy = drone_pos.x_val, drone_pos.y_val
        objs = []
        for (gx, gy), cell in self.grid_map.items():
            if cell['conf'] < 60: continue
            label = max(cell['votes'], key=cell['votes'].get)
            if label == "road": continue
            wx, wy = gx * GRID_SIZE, gy * GRID_SIZE
            dist = math.sqrt((wx - dx)**2 + (wy - dy)**2)
            
            # [修正 1] 物件列表過濾：只列出 10 米內的物件
            if dist > 10.0: continue 
            
            angle = (math.degrees(math.atan2(wy - dy, wx - dx) - drone_yaw) + 180) % 360 - 180
            clock = "12"
            if 15 <= angle < 45: clock = "1-2"
            elif 45 <= angle < 135: clock = "3"
            elif -45 < angle <= -15: clock = "10-11"
            elif -135 < angle <= -45: clock = "9"
            elif abs(angle) >= 135: clock = "6"
            objs.append(f"- {label.upper()} at {dist:.1f}m ({clock} o'clock)")
            
        objs.sort(key=lambda x: float(x.split('at ')[1].split('m')[0]))
        objs_str = "\n".join(objs[:6]) if objs else "None (Area Clear)"
        
        confirmed_str = ""
        if confirmed_targets:
            confirmed_list = []
            for label, positions in confirmed_targets.items():
                for pos, _ in positions:
                    dist = math.sqrt((pos[0] - dx)**2 + (pos[1] - dy)**2)
                    confirmed_list.append(f"  * {label.upper()} confirmed at ({pos[0]:.1f}, {pos[1]:.1f}), {dist:.1f}m away")
            if confirmed_list:
                confirmed_str = "\n- Confirmed Targets:\n" + "\n".join(confirmed_list[:5])
        
        # [修正 1] 強制狀態判斷：遠距離誤判修正
        dist_L, dist_C, dist_R = gaps
        BLOCK_DIST = 4.0      
        WALL_CHECK_DIST = 6.0 
        
        space_status = "Open"
        suggestion = ""

        # 直接用距離決定文字，避免 LLM 對遠處物件產生恐慌
        if dist_C > 8.0:
             space_status = f"PATH CLEAR (Front {dist_C:.1f}m)"
             suggestion = "Safe to MOVE_FORWARD."
        elif dist_C > BLOCK_DIST:
             space_status = f"APPROACHING OBSTACLE (Front {dist_C:.1f}m)"
             suggestion = "Caution, but can still MOVE_FORWARD."
        else:
            # 只有真的小於 BLOCK_DIST (4.0m) 才顯示 BLOCKED
            is_wall_left = dist_L < WALL_CHECK_DIST
            is_wall_right = dist_R < WALL_CHECK_DIST
            
            if is_wall_left and is_wall_right:
                space_status = "LARGE WALL / DEAD END"
                suggestion = "Both sides blocked. Suggest ROTATE (Turn)."
            elif not is_wall_left:
                space_status = f"OBSTACLE ({dist_C:.1f}m), LEFT OPEN"
                suggestion = "Left side is clear. Suggest STRAFE_LEFT."
            elif not is_wall_right:
                space_status = f"OBSTACLE ({dist_C:.1f}m), RIGHT OPEN"
                suggestion = "Right side is clear. Suggest STRAFE_RIGHT."
            else:
                space_status = "BLOCKED"
                suggestion = "Suggest ROTATE."
        
        prompt = (
            f"Context:\n"
            f"- Scene: {scene_context}\n"
            f"- Exploration Status: {exploration_status}\n"
            f"- Spatial: Front:{dist_C:.1f}m, Left:{dist_L:.1f}m, Right:{dist_R:.1f}m.\n"
            f"- Status: {space_status} {suggestion}\n"  # 加入建議
            f"- Objects: {objs_str}{confirmed_str}\n"
            f"- Previous Action: {last_action}\n"
            f"\nReason about the situation. Follow the Strategy to decide action."
        )
        return prompt
    
    def get_visualization(self, pos, yaw, path_history=[], confirmed_targets={}):
        img = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS, 3), dtype=np.uint8)
        cx, cy = MAP_SIZE_PIXELS // 2, MAP_SIZE_PIXELS // 2
        dx, dy = pos.x_val, pos.y_val
        
        for (gx, gy), cell in self.grid_map.items():
            label = max(cell['votes'], key=cell['votes'].get)
            if label == "road": continue
            wx, wy = gx * GRID_SIZE, gy * GRID_SIZE
            px, py = int(cx + (wy - dy) / METERS_PER_PIXEL), int(cy - (wx - dx) / METERS_PER_PIXEL)
            if 0 <= px < MAP_SIZE_PIXELS and 0 <= py < MAP_SIZE_PIXELS:
                col = self.COLORS.get(label, self.COLORS["unknown"])
                cv2.rectangle(img, (px-2, py-2), (px+2, py+2), col, -1)

        for label, positions in confirmed_targets.items():
            col = self.COLORS.get(label, (255, 255, 0))
            for world_pos, _ in positions:
                wx, wy = world_pos
                px = int(cx + (wy - dy) / METERS_PER_PIXEL)
                py = int(cy - (wx - dx) / METERS_PER_PIXEL)
                if 0 <= px < MAP_SIZE_PIXELS and 0 <= py < MAP_SIZE_PIXELS:
                    cv2.circle(img, (px, py), 12, col, 2)
                    cv2.circle(img, (px, py), 8, (255, 255, 0), -1)
                    cv2.putText(img, f"{label.upper()}!", (px + 15, py), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    coord_text = f"({wx:.1f}, {wy:.1f})"
                    cv2.putText(img, coord_text, (px + 15, py + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if len(path_history) > 1:
            pts = []
            for (hx, hy) in path_history:
                px = int(cx + (hy - dy) / METERS_PER_PIXEL)
                py = int(cy - (hx - dx) / METERS_PER_PIXEL)
                pts.append([px, py])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, (0, 0, 255), 1)

        cv2.circle(img, (cx, cy), 8, (0, 255, 255), -1)
        end_x = int(cx + 30 * math.sin(yaw))
        end_y = int(cy - 30 * math.cos(yaw))
        cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (0, 255, 255), 2)
        return img

def get_fused_perception(depth, yolo_res, w, h, alt):
    points = []
    boxes = [{'roi': box.xyxy[0].cpu().numpy(), 'label': yolo_res[0].names[int(box.cls[0])].lower()} 
             for box in yolo_res[0].boxes]
    fx, fy = w / (2 * math.tan(math.radians(45))), h / (2 * math.tan(math.radians(45)))
    cx, cy = w / 2, h / 2
    step = 4
    for v in range(int(h*0.2), int(h*0.8), step):
        for u in range(0, w, step):
            d = depth[v, u]
            if d < 0.5 or d > 40: continue
            is_inside = False
            lbl = "unknown"
            for b in boxes:
                if b['roi'][0]<=u<=b['roi'][2] and b['roi'][1]<=v<=b['roi'][3]:
                    is_inside = True; lbl = b['label']; break
            limit = abs(alt) - (0.5 if is_inside else 2.5)
            y_cam = (v - cy) * d / fy
            if y_cam > limit: continue 
            points.append((d, (u-cx)*d/fx, lbl))
    return points

def get_depth_gaps(depth_img_float):
    h, w = depth_img_float.shape
    # [MODIFIED] 修改寬度比例，只看中間 20% (0.4~0.6)
    strip_L = depth_img_float[int(h*0.2):int(h*0.7), 0:int(w*0.4)]
    strip_C = depth_img_float[int(h*0.2):int(h*0.7), int(w*0.4):int(w*0.6)]
    strip_R = depth_img_float[int(h*0.2):int(h*0.7), int(w*0.6):w]
    gaps = []
    for s in [strip_L, strip_C, strip_R]:
        val = np.percentile(s, 1) if s.size > 0 else 50.0 
        gaps.append(val)
    return gaps 

# [NEW] 動態計算平移距離函式 (含脫離緩衝與防呆)
# ==========================================
def calculate_dynamic_strafe_distance(depth_img, direction, current_obstacle_dist, fov=90.0):
    """
    依據深度圖計算閃避障礙物所需的最小平移距離
    :param depth_img: 深度影像 (2D array)
    :param direction: "STRAFE_LEFT" 或 "STRAFE_RIGHT"
    :param current_obstacle_dist: 目前前方障礙物的距離 (dist_C)
    :param fov: 攝影機水平視角 (預設 90 度)
    :return: 建議的平移距離 (meters)
    """
    h, w = depth_img.shape
    cx = w / 2
    fx = w / (2 * math.tan(math.radians(fov / 2)))
    
    scan_strip = depth_img[int(h*0.4):int(h*0.6), :]
    clear_threshold = max(current_obstacle_dist + 2.0, 5.0)
    
    edge_pixel_index = -1
    
    if direction == "STRAFE_LEFT":
        # 向左掃描
        for u in range(int(w/2), -1, -1):
            col_depth = np.median(scan_strip[:, u])
            if col_depth > clear_threshold:
                edge_pixel_index = u
                break
    elif direction == "STRAFE_RIGHT":
        # 向右掃描
        for u in range(int(w/2), w):
            col_depth = np.median(scan_strip[:, u])
            if col_depth > clear_threshold:
                edge_pixel_index = u
                break
    
    # [防呆機制] 如果找不到邊緣 (整片都是牆)，回傳極短距離試探
    if edge_pixel_index == -1:
        print(f"⚠️ 無法偵測障礙物邊緣(判定為牆)，強制縮小平移幅度")
        return 1.5 
        
    # 計算橫向距離
    pixel_offset = abs(edge_pixel_index - cx)
    lateral_dist = current_obstacle_dist * (pixel_offset / fx)
    
    # [修正 2] 加大脫離視野中心的力道
    # 讓物體不只離開中心點，而是離開中間 35% 的區域，確保不會再被誤判為 Blocked
    fov_width_at_dist = 2 * current_obstacle_dist * math.tan(math.radians(fov/2))
    exit_buffer = fov_width_at_dist * 0.35 # 改為 0.35
    
    # 最終距離 = 邊緣距離 + 脫離緩衝 + 機身安全半徑
    final_dist = lateral_dist + exit_buffer + 0.5
    
    # [修正 2] 稍微放寬最大平移限制 (原本 4.0 改為 5.0)
    final_dist = max(1.5, min(final_dist, 5.0))
    
    print(f"📏 動態計算: 障礙距 {current_obstacle_dist:.1f}m, 需橫移 {final_dist:.1f}m (含脫離緩衝)")
    return final_dist

# ==========================================
# 主程式
# ==========================================
def main():
    config_ui = MissionConfigDialog()
    user_classes, user_prompt = config_ui.get_input()
    
    print(f"啟動: 目標={user_classes}")
    print("模式: 具體動作控制 + YOLO 確認追蹤")
    
    yolo = YOLOWorld("yolov8l-world.pt")
    yolo.set_classes(user_classes)
    
    scene_net = FastSceneClassifier()
    memory = RobustMemorySystem()
    path_memory = PathMemory() 
    planner = GapSeekingPlanner() 
    coverage = CoverageMap()
    yolo_tracker = YOLOTargetTracker(YOLO_CONFIRMATION_FRAMES, YOLO_TRACKING_DISTANCE) 
    
    brain_thread = AsyncLLMBrain(api_key=GROQ_API_KEY, system_prompt=user_prompt)
    brain_thread.start()
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    client.moveToZAsync(FLIGHT_ALTITUDE, 3).join()
    
    action_executor = ActionExecutor(client) 
    
    try:
        while True:
            # 1. 感測
            res = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
            ])
            if len(res) < 2: continue
            
            rgb = np.frombuffer(res[0].image_data_uint8, dtype=np.uint8).reshape(res[0].height, res[0].width, 3)
            depth = np.array(res[1].image_data_float, dtype=np.float32).reshape(res[1].height, res[1].width)
            
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            q = state.kinematics_estimated.orientation
            drone_yaw = math.atan2(2*(q.w_val*q.z_val + q.x_val*q.y_val), 1-2*(q.y_val**2 + q.z_val**2))

            # 2. YOLO 檢測 + 追蹤
            yolo_out = yolo.predict(rgb, conf=YOLO_CONFIDENCE, verbose=False)
            
            yolo_detections = []
            tracking_detections = [] 
            
            if len(yolo_out) > 0:
                fx = res[0].width / (2 * math.tan(math.radians(45)))
                cx, cy = res[0].width / 2, res[0].height / 2
                
                for box in yolo_out[0].boxes:
                    cls_id = int(box.cls[0])
                    label = yolo_out[0].names[cls_id].lower()
                    roi = box.xyxy[0].cpu().numpy()
                    yolo_detections.append({'label': label, 'roi': roi})
                    
                    box_cx = (roi[0] + roi[2]) / 2
                    box_cy = (roi[1] + roi[3]) / 2
                    d = depth[int(box_cy), int(box_cx)] if 0 <= int(box_cy) < depth.shape[0] and 0 <= int(box_cx) < depth.shape[1] else 5.0
                    
                    lat = (box_cx - cx) * d / fx
                    wx = pos.x_val + (d * math.cos(drone_yaw) - lat * math.sin(drone_yaw))
                    wy = pos.y_val + (d * math.sin(drone_yaw) + lat * math.cos(drone_yaw))
                    
                    tracking_detections.append({
                        'label': label,
                        'world_pos': (wx, wy)
                    })
            
            yolo_tracker.update(tracking_detections, pos, drone_yaw)
            confirmed_targets = yolo_tracker.get_confirmed_targets()

            scene_desc = scene_net.predict(rgb)
            points = get_fused_perception(depth, yolo_out, res[0].width, res[0].height, FLIGHT_ALTITUDE)
            memory.update(pos, drone_yaw, points)
            path_memory.update_position(pos.x_val, pos.y_val, time.time())
            coverage.update(pos.x_val, pos.y_val)
            
            gaps = get_depth_gaps(depth)

            # 3. 動作執行檢查
            action_completed = action_executor.check_completion(pos, drone_yaw)
            
            # 4. LLM 決策
            if action_completed and not GLOBAL_STATE["new_data_available"]:
                last_act = GLOBAL_STATE.get("current_action", "None")
                exp_status = coverage.get_exploration_stats()
                
                prompt_text = memory.generate_llm_prompt(
                    pos, drone_yaw, scene_desc, gaps, last_act, exp_status, confirmed_targets
                )
                
                GLOBAL_STATE["perception_text"] = prompt_text
                GLOBAL_STATE["new_data_available"] = True
            
            # 5. 控制融合
            control_msg = "AI FLIGHT"
            collision_info = client.simGetCollisionInfo()
            
            # [MODIFIED] 修改控制邏輯，加入動態距離計算
            # -----------------------------------------------------------
            if action_completed and GLOBAL_STATE["current_action"] and not action_executor.is_executing:
                llm_action = GLOBAL_STATE["current_action"]
                action_params = GLOBAL_STATE["current_action_params"]
                
                # === 動態距離計算邏輯 ===
                if llm_action in ["STRAFE_LEFT", "STRAFE_RIGHT"]:
                    current_obs_dist = gaps[1] 
                    if current_obs_dist < 5.0:
                        dynamic_dist = calculate_dynamic_strafe_distance(
                            depth,          # 傳入深度圖
                            llm_action,     # 傳入方向
                            current_obs_dist # 傳入障礙物距離
                        )
                        action_params['distance'] = dynamic_dist
                        print(f"🤖 系統介入: 將平移距離修正為 {dynamic_dist:.1f}m")
                # ===============================
                
                action_executor.start_action(llm_action, action_params, pos, drone_yaw)
            # -----------------------------------------------------------
            
            if action_executor.is_executing:
                vx, vy, yaw_rate = action_executor.get_velocity_command(drone_yaw)
                control_msg = f"EXECUTING: {action_executor.current_action}"
            else:
                path_repulsion = path_memory.get_repulsion_force(pos.x_val, pos.y_val, time.time())
                vx, vy, yaw_rate, vis_debug = planner.calculate_global_velocity(
                    0.3, 0, gaps, drone_yaw, pos.x_val, pos.y_val, coverage,
                    path_repulsion=path_repulsion,
                    yolo_detections=yolo_detections,
                    img_width=res[0].width
                )
                control_msg = "STANDBY"
            
            # 碰撞保護
            if collision_info.has_collided:
                print(f"⚠️ 撞擊偵測! 執行緊急倒車!")
                vx = -0.8 * math.cos(drone_yaw)
                vy = -0.8 * math.sin(drone_yaw)
                yaw_rate = 0.0 
                control_msg = "COLLISION RECOVERY"
                action_executor.is_executing = False 
            elif gaps[1] < EMERGENCY_DIST:
                back_vx = -0.5 * math.cos(drone_yaw)
                back_vy = -0.5 * math.sin(drone_yaw)
                vx, vy = back_vx, back_vy
                control_msg = "REFLEX BRAKE!"
            
            client.moveByVelocityAsync(vx, vy, 0, duration=5.0, 
                                       drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                       yaw_mode=airsim.YawMode(True, yaw_rate))
            
            # 6. 視覺化
            yolo_plot = yolo_out[0].plot()
            
            for label, positions in confirmed_targets.items():
                for world_pos, _ in positions:
                    dist = math.sqrt((world_pos[0] - pos.x_val)**2 + (world_pos[1] - pos.y_val)**2)
                    if dist < 20: 
                        cv2.putText(yolo_plot, f"CONFIRMED: {label.upper()}", (50, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            llm_action_display = f"{GLOBAL_STATE.get('current_action', 'N/A')} {GLOBAL_STATE.get('current_action_params', {})}"
            cv2.putText(rgb, f"Scene: {scene_desc}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(rgb, f"LLM: {llm_action_display}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.putText(rgb, f"Status: {control_msg}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(rgb, f"Confirmed: {sum(len(v) for v in confirmed_targets.values())}", (10,120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            cv2.imshow("Drone Eye", yolo_plot)
            mem_img = memory.get_visualization(pos, drone_yaw, list(path_memory.position_history), confirmed_targets)
            cv2.imshow("Brain Memory", mem_img)
            
            depth_disp = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_color = cv2.applyColorMap(np.uint8(depth_disp), cv2.COLORMAP_JET)
            cv2.imshow("Depth Space", depth_color)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    finally:
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()