"""
visualize_l3.py - Layer 3 APF 避障軌跡視覺化
=============================================
利用 matplotlib 繪製 APF 獨立運行下的避障軌跡，
展示 20% 正交微擾力 (Orthogonal Breaking Force) 如何打破重線局部極小值死鎖。
"""
import matplotlib.pyplot as plt
from layer3_standalone import Layer3Controller, Vector3
import numpy as np

def run_simulation(controller, target, obs, steps=500):
    controller.target = target
    controller.obstacles = obs
    path = []
    
    for _ in range(steps):
        p, v = controller.step()
        path.append((p.x, p.y))
        if p.dist(target) < 0.5:
            break
            
    return path

def plot_apf_path():
    # 建立控制器
    controller = Layer3Controller(hz=50)
    
    # 測試情境: 完美的重線阻擋 (0,0) -> (10,0)，障礙物在 (5,0)
    start_pos = Vector3(0, 0, 0)
    target_pos = Vector3(10, 0, 0)
    obstacle = {"x": 5.0, "y": 0.0, "z": 0.0}
    
    controller.pos = Vector3(start_pos.x, start_pos.y, start_pos.z)
    
    print("\n🚀 [Layer 3 Visualizer] 開始模擬 APF 避障...")
    path = run_simulation(controller, target_pos, [obstacle])
    
    # 轉換路徑資料
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    
    # 開始繪圖
    plt.figure(figsize=(10, 6))
    
    # 繪製起點與終點
    plt.plot(start_pos.x, start_pos.y, 'go', markersize=10, label='Start (0, 0)')
    plt.plot(target_pos.x, target_pos.y, 'b*', markersize=15, label='Target (10, 0)')
    
    # 繪製障礙物與斥力影響範圍 (rho_0 = 2.5m)
    circle = plt.Circle((obstacle['x'], obstacle['y']), 2.5, color='r', fill=True, alpha=0.2, label='Obstacle Influence (2.5m)')
    plt.gca().add_patch(circle)
    plt.plot(obstacle['x'], obstacle['y'], 'rX', markersize=12, label='Obstacle Core (5, 0)')
    
    # 繪製無人機軌跡
    plt.plot(xs, ys, 'k--', linewidth=2, label='Drone Path (APF with Orthogonal Force)')
    
    # 加入箭頭標示方向
    for i in range(0, len(xs)-1, max(1, len(xs)//10)):
        plt.annotate('', xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.title("Layer 3: APF Obstacle Avoidance Path (Local Minima Breakthrough)", fontsize=14, fontweight='bold')
    plt.xlabel("X (m)", fontsize=12)
    plt.ylabel("Y (m)", fontsize=12)
    plt.legend(loc='upper right')
    plt.axis([-2, 12, -4, 4])
    
    print("✅ [Layer 3 Visualizer] 模擬完成，開啟視覺化視窗！")
    
    # 直接顯示出來給 User 看
    plt.show()

if __name__ == "__main__":
    plot_apf_path()
