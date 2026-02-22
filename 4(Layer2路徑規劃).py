import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
import random

# ==========================================
# 參數設定
# ==========================================
MAP_PATH = r"C:\Users\Ethan\Desktop\畢業專題3\新測試程式碼 2\airsim模擬器測試\ollama版本\meta-llama\llama-3.1-8b-instant V4\「視覺化記憶驗證實驗\整合LLM推論\Map_RT.EXR"
DILATION_SIZE = 5   # 安全緩衝區
STEP_SIZE = 10      # A* 步長

# ==========================================
# 1. 地圖處理模組
# ==========================================
def load_and_process_map(image_path):
    if not os.path.exists(image_path):
        print(f"❌ 找不到路徑: {image_path}")
        return None, None

    with open(image_path, 'rb') as f:
        buffer = f.read()
    img = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_UNCHANGED)
    
    if img is None: return None, None
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # 影像增強
    img = np.nan_to_num(img)
    norm_img = np.power((img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5), 0.45)
    rgb_map = (np.clip(norm_img, 0, 1) * 255).astype(np.uint8)

    print("🔄 正在生成導航地圖...")
    h, w = rgb_map.shape[:2]
    binary_map = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            b, g, r = rgb_map[y, x]
            brightness = int(b) + int(g) + int(r)
            # 判斷草地
            is_grass = (g >= r and g >= b) and (brightness > 10)
            
            if is_grass:
                binary_map[y, x] = 0   # 安全
            else:
                binary_map[y, x] = 255 # 障礙

    # 膨脹處理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DILATION_SIZE, DILATION_SIZE))
    dilated_map = cv2.dilate(binary_map, kernel)
    
    return rgb_map, dilated_map

# ==========================================
# 2. 【核心新增】連通性分析與智慧選點
# ==========================================
def find_largest_safe_zone_and_points(obstacle_map):
    print("🔍 正在進行連通性分析 (Connectivity Analysis)...")
    
    h, w = obstacle_map.shape
    
    # 1. 製作「安全區遮罩」 (反轉障礙圖：安全=255, 障礙=0)
    # 這樣才能用 connectedComponents 找「白色(安全)」的區塊
    safe_mask = cv2.bitwise_not(obstacle_map)
    
    # 2. 找出所有連通區域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(safe_mask, connectivity=8)
    
    print(f"   -> 偵測到 {num_labels - 1} 個獨立的安全區域")

    # 3. 找出面積最大的區域 (排除背景 0)
    max_area = 0
    max_label = 0
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_label = i
            
    print(f"   -> 鎖定最大主飛行區域 (Label {max_label}, Area: {max_area} pixels)")
    
    # 4. 在這個最大區域內挑選起點與終點
    # 建立只包含主區域的遮罩
    main_zone_mask = (labels == max_label).astype(np.uint8) * 255
    
    # 找出所有屬於主區域的座標點
    y_indices, x_indices = np.where(main_zone_mask == 255)
    
    if len(x_indices) == 0:
        return None, None
        
    # 策略：挑選最左上角的點當起點，最右下角的點當終點
    # 為了避免太貼邊，我們取百分位數 (10% 和 90% 的位置)
    
    # 將座標排序
    coords = list(zip(y_indices, x_indices))
    coords.sort(key=lambda p: p[0] + p[1]) # 依據 (y+x) 排序 -> 左上到右下
    
    start_idx = int(len(coords) * 0.05) # 前 5% 的位置
    goal_idx = int(len(coords) * 0.95)  # 後 95% 的位置
    
    auto_start = coords[start_idx]
    auto_goal = coords[goal_idx]
    
    print(f"✅ 自動鎖定主區域內的起點 {auto_start} 與 終點 {auto_goal}")
    return auto_start, auto_goal

# ==========================================
# 3. A* 路徑規劃
# ==========================================
class AStarPlanner:
    def __init__(self, obstacle_map, step_size=10):
        self.obstacle_map = obstacle_map
        self.h, self.w = obstacle_map.shape
        self.step_size = step_size
        self.motions = [
            (0, step_size), (0, -step_size), (step_size, 0), (-step_size, 0),
            (step_size, step_size), (step_size, -step_size), 
            (-step_size, step_size), (-step_size, -step_size)
        ]

    def heuristic(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def is_safe(self, node):
        y, x = node
        if not (0 <= x < self.w and 0 <= y < self.h): return False
        if self.obstacle_map[int(y), int(x)] > 127: return False
        return True

    def search(self, start, goal):
        print(f"📡 A* 開始運算路徑...")
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        iter_count = 0
        while open_list:
            _, current = heapq.heappop(open_list)
            iter_count += 1
            if iter_count % 10000 == 0:
                print(f"   ...已搜索 {iter_count} 節點...")

            if self.heuristic(current, goal) < self.step_size:
                print("✅ 路徑計算成功！")
                return self.reconstruct_path(came_from, start, current)
            
            for dy, dx in self.motions:
                next_node = (current[0] + dy, current[1] + dx)
                new_cost = cost_so_far[current] + np.sqrt(dy**2 + dx**2)
                
                if self.is_safe(next_node):
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + self.heuristic(goal, next_node)
                        heapq.heappush(open_list, (priority, next_node))
                        came_from[next_node] = current
        return None

    def reconstruct_path(self, came_from, start, current):
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    rgb_map, nav_map = load_and_process_map(MAP_PATH)
    
    if nav_map is not None:
        # 使用【連通性分析】自動找點，而不是手動輸入
        start_pos, goal_pos = find_largest_safe_zone_and_points(nav_map)

        if start_pos and goal_pos:
            planner = AStarPlanner(nav_map, step_size=STEP_SIZE)
            waypoints = planner.search(start_pos, goal_pos)
            
            print("🎨 正在繪製結果圖...")
            result_img = cv2.cvtColor(rgb_map, cv2.COLOR_BGR2RGB)
            
            # 畫起點終點
            cv2.circle(result_img, (start_pos[1], start_pos[0]), 50, (255, 0, 0), -1)
            cv2.circle(result_img, (goal_pos[1], goal_pos[0]), 50, (0, 0, 255), -1)

            title_text = "Layer 2 Robust Path Planning"
            
            if waypoints:
                for i in range(len(waypoints) - 1):
                    pt1 = (waypoints[i][1], waypoints[i][0])   
                    pt2 = (waypoints[i+1][1], waypoints[i+1][0])
                    cv2.line(result_img, pt1, pt2, (0, 255, 0), 8) # 綠色路徑
                title_text += ": SUCCESS"
            else:
                title_text += ": FAILURE"
                print("❌ 即使在主區域內也找不到路徑，請檢查地圖是否破碎嚴重。")

            plt.figure(figsize=(12, 12))
            plt.imshow(result_img)
            plt.title(title_text, fontsize=16, fontweight='bold', color='green' if waypoints else 'red')
            plt.axis("off")
            
            plt.savefig("layer2_robust_result.png")
            print("💾 結果圖已儲存為 layer2_robust_result.png")
            plt.show()
        else:
            print("❌ 地圖上找不到夠大的安全區域！請檢查 Map_RT.EXR 是否過暗或全黑。")
    else:
        print("地圖讀取失敗。")