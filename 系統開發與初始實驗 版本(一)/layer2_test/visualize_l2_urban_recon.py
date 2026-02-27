"""
visualize_l2_urban_recon.py - 1 平方公里都市偵蒐視覺化
======================================================
展示 Layer 2 針對大面積目標自動生成 Macro-Sweep 網格路徑，
並展示電量審查機制如何攔截超出續航力的荒謬任務。
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from layer2_standalone import Layer2BehaviorPlanner, Layer1Decision, Waypoint

def plot_urban_recon():
    # 建立 1km x 1km 環境
    env = {
        "Urban_1KM": {
            "type": "Building_Zone", "danger_level": "High",
            "boundary": { "north": 1000.0, "south": 0.0, "east": 1000.0, "west": 0.0 }
        }
    }
    with open("urban_env_test.json", "w", encoding="utf-8") as f:
        json.dump(env, f)
        
    planner = Layer2BehaviorPlanner(env_path="urban_env_test.json")
    
    # 模擬 LLM 產生的簡略邊界點
    decision_dict = {
        "mission_id": "RECON_01",
        "command": "SEARCH",
        "target_area": "Urban_1KM",
        "waypoints": [{"x":0, "y":0, "z":-50}, {"x":1000, "y":1000, "z":-50}],
        "altitude": -50.0,
        "speed": 10.0,  # 提高速度，不一定飛得完
        "timeout_sec": 3600.0,
        "priority": 9
    }
    decision = Layer1Decision.from_json(decision_dict)
    
    # 模擬 process_layer1_decision 中的前段邏輯
    raw_waypoints = planner.planner.from_layer1_decision(decision)
    
    # 手動觸發巨集掃描
    tgt_data = planner.geofence.areas.get(decision.target_area)
    b = tgt_data["boundary"]
    
    print("\n🌍 [Visualizer] 觸發宏觀 S 型網格生成 (掃描寬度: 50m，為了繪圖不要太密)")
    macro_waypoints = planner.planner.generate_macro_sweep(
        west=b["west"], east=b["east"], 
        south=b["south"], north=b["north"], 
        altitude=decision.altitude,
        scan_width=50.0  # 為了圖表清楚，設寬一點
    )
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(10, 10))
    rect = patches.Rectangle((0, 0), 1000, 1000, linewidth=2, edgecolor='firebrick', facecolor='firebrick', alpha=0.3)
    ax.add_patch(rect)
    ax.text(500, 500, "Urban_1KM (1,000,000 m^2)", ha='center', va='center', fontsize=16, color='darkred', fontweight='bold')
    
    xs = [w.x for w in macro_waypoints]
    ys = [w.y for w in macro_waypoints]
    
    ax.plot(xs, ys, 'b--', linewidth=1.5, label='Macro-Sweep Pattern')
    ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start')
    ax.plot(xs[-1], ys[-1], 'ro', markersize=10, label='End')
    
    total_dist = planner.planner.compute_total_distance(macro_waypoints)
    est_time = total_dist / decision.speed
    
    title = f"Layer 2 Macro-Sweep for 1 sq km Urban Recon\nTotal Dist: {total_dist/1000:.1f} km | Est. Time: {est_time/60:.1f} mins"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right')
    
    print(f"✅ 生成完畢。總航點數: {len(macro_waypoints)}")
    print(f"🔋 [Battery Validation] 若最大續航 30Mins，這個任務耗時 {est_time/60:.1f}Mins 會在代碼中被攔截 Exception！")
    
    plt.savefig("l2_urban_recon_macro.png")
    print("已儲存為 l2_urban_recon_macro.png")
    plt.show()

if __name__ == "__main__":
    plot_urban_recon()
