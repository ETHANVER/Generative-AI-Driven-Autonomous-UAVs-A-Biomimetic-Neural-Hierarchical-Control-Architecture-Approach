"""
visualize_l2.py - Layer 2 Geofencing 與航點視覺化
===================================================
讀取 environment.json，繪製地理圍欄分區圖，
並模擬 Layer 1 傳來的航點，展示 Geofence 如何攔截非法路徑
以及 PathPlanner 如何自動安插安全爬升點 (AUTO_CLIMB_POINT)。
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from layer2_standalone import Layer2BehaviorPlanner, Layer1Decision, Waypoint

def draw_environment(ax, env_path="environment.json"):
    with open(env_path, 'r', encoding='utf-8') as f:
        env_data = json.load(f)
        
    # 色彩對應表
    color_map = {
        "Forest": "forestgreen",
        "Building_Zone": "firebrick",
        "Open_Field": "gold",
        "Vegetation": "yellowgreen"
    }

    print("\n🗺️ [Layer 2 Visualizer] 載入環境地圖...")
    for zone_id, zone_info in env_data.items():
        if "global" in zone_id.lower(): continue # 略過全域摘要
        
        b = zone_info.get("boundary", {})
        z_type = zone_info.get("type", "Unknown")
        danger = zone_info.get("danger_level", "Low")
        
        west, east, south, north = b.get("west"), b.get("east"), b.get("south"), b.get("north")
        
        # 繪製矩形
        width = east - west
        height = north - south
        
        color = color_map.get(z_type, "gray")
        alpha = 0.4 if danger != "High" else 0.6
        
        rect = patches.Rectangle((west, south), width, height, linewidth=2, edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        
        # 標籤
        center_x = west + width/2
        center_y = south + height/2
        label = f"{zone_id}\n({z_type})\nDanger: {danger}"
        ax.text(center_x, center_y, label, ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

def simulate_and_draw_path(ax):
    # 初始化 Layer 2 Planner
    planner = Layer2BehaviorPlanner(env_path="environment.json")
    
    # 建立一個測試決策：起點 -> 穿越森林 -> 飛入禁航區
    print("\n🚀 [Layer 2 Visualizer] 模擬 Layer 1 決策路徑生成...")
    raw_waypoints = [
        Waypoint(x=10, y=10, z=-10, label="Start"),          # Area1
        Waypoint(x=30, y=30, z=-15, label="ForestSweep"),    # Area1
        Waypoint(x=60, y=60, z=-30, label="HighAltClimb"),   # Area1 到 Area2 邊境 (高度落差大)
        Waypoint(x=-10, y=-10, z=-10, label="HackedPoint")   # 故意飛入 Area2 (高危建築區，預期被 Geofence 攔截)
    ]
    
    xs, ys = [], []
    valid_waypoints = []
    
    # 手動展示 Geofence 攔截邏輯
    for wp in raw_waypoints:
        is_safe, msg = planner.geofence.check(wp.x, wp.y)
        if is_safe:
            valid_waypoints.append(wp)
            xs.append(wp.x)
            ys.append(wp.y)
            ax.plot(wp.x, wp.y, 'go', markersize=8)
            ax.text(wp.x+2, wp.y+2, f"{wp.label}\n(z={wp.z})", color='green', fontsize=8)
        else:
            print(f"🛡️  [Geofence 攔截] 航點 {wp.label} ({wp.x}, {wp.y}) -> {msg}")
            # 畫一個紅色的叉叉
            ax.plot(wp.x, wp.y, 'rX', markersize=12)
            ax.text(wp.x+2, wp.y-4, f"BLOCKED:\n{wp.label}", color='red', fontsize=8, fontweight='bold')
            # 飛到這就截斷
            break
            
    # 展示 PathPlanner 的插值邏輯 (安全高度點)
    print("\n🛠️ [PathPlanner] 執行插值平滑優化...")
    final_path = planner.planner.insert_safe_altitude(valid_waypoints, safe_z=-20.0, climb_threshold=10.0)
    
    fxs = [w.x for w in final_path]
    fys = [w.y for w in final_path]
    
    # 畫出最終路徑
    ax.plot(fxs, fys, 'b--', linewidth=2.5, label="Optimized Safe Path")
    
    # 標示出安插的點
    for w in final_path:
        if w.label == "AUTO_CLIMB_POINT":
            ax.plot(w.x, w.y, 'y^', markersize=10)
            ax.text(w.x-10, w.y+3, f"AUTO_CLIMB\n(z={w.z})", color='goldenrod', fontsize=8, fontweight='bold')

def plot_l2_environment():
    fig, ax = plt.subplots(figsize=(12, 12))
    
    draw_environment(ax, "environment.json")
    simulate_and_draw_path(ax)
    
    ax.set_xlim(-50, 150)
    ax.set_ylim(-50, 150)
    ax.set_title("Layer 2: Geofencing & Code-as-Planner Visualization\n(Forest Patrol + Intrusion Block + Auto Climb)", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # 自定義 Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='b', ls='--', lw=2.5),
        Line2D([0], [0], marker='o', color='g', linestyle='None'),
        Line2D([0], [0], marker='^', color='y', linestyle='None'),
        Line2D([0], [0], marker='X', color='r', linestyle='None')
    ]
    ax.legend(custom_lines, ['Execution Path', 'Valid Waypoint', 'Auto Climb Point (Interpolated)', 'Blocked/Hallucination (Geofence)'], loc='lower right')
    
    print("✅ [Layer 2 Visualizer] 模擬完成，開啟視覺化視窗！")
    plt.show()

if __name__ == "__main__":
    plot_l2_environment()
