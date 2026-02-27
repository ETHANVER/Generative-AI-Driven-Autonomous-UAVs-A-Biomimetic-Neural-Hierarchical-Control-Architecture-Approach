"""
visualize_l3_ushape.py - Layer 3 APF 沿牆逃脫 (U-Shape Escape) 視覺化
========================================================================
展示新的 APF 引擎如何透過點積偵測「本地死鎖 (Local Minima)」並啟動沿牆模式，
成功帶領無人機逃出 U 字型大樓的包圍。
"""
import matplotlib.pyplot as plt
from layer3_standalone import Layer3Controller, Vector3
import numpy as np

def run_simulation(controller, target, obs, steps=1000):
    controller.target = target
    controller.obstacles = obs
    path = []
    
    for _ in range(steps):
        p, v = controller.step()
        path.append((p.x, p.y))
        if p.dist(target) < 0.5:
            break
            
    return path

def plot_ushape_escape():
    # 建立控制器
    controller = Layer3Controller(hz=50)
    
    # 起點與終點
    start_pos = Vector3(0, 0, 0)
    target_pos = Vector3(20, 0, 0)
    
    # 建立一個完美的 ⼕ 字型 (U-Shape) 建築物牆面，擋在中間 (x=10)
    # 用多個密集點狀障礙物組成牆面
    obstacles = []
    # 主牆面 (直立)
    for y in np.linspace(-4, 4, 9):
        obstacles.append({"x": 10.0, "y": y, "z": 0.0})
    # 側邊擋牆 (水平，向前延伸形成 U 型包圍)
    for x in np.linspace(7, 10, 4):
        obstacles.append({"x": x, "y": 4.0, "z": 0.0})
        obstacles.append({"x": x, "y": -4.0, "z": 0.0})
    
    controller.pos = Vector3(start_pos.x, start_pos.y, start_pos.z)
    
    print("\n🚀 [Layer 3 Visualizer] 開始模擬 U 型建築強制逃脫...")
    path = run_simulation(controller, target_pos, obstacles)
    
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    
    # 開始繪圖
    plt.figure(figsize=(10, 6))
    
    plt.plot(start_pos.x, start_pos.y, 'go', markersize=10, label='Start (0, 0)')
    plt.plot(target_pos.x, target_pos.y, 'b*', markersize=15, label='Target (20, 0)')
    
    # 繪製障礙牆
    obs_xs = [o['x'] for o in obstacles]
    obs_ys = [o['y'] for o in obstacles]
    plt.plot(obs_xs, obs_ys, 'rs', markersize=8, label='U-Shape Building Wall')
    
    # 繪製影響半徑
    for o in obstacles:
        circle = plt.Circle((o['x'], o['y']), 2.5, color='r', fill=True, alpha=0.05)
        plt.gca().add_patch(circle)
    
    # 繪製無人機軌跡
    plt.plot(xs, ys, 'k-', linewidth=3, label='Drone Path (Wall-Following Escape)')
    
    # 加入重點箭頭
    for i in range(0, len(xs)-1, max(1, len(xs)//15)):
        plt.annotate('', xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                     arrowprops=dict(arrowstyle="->", color="orange", lw=2))

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("Layer 3: U-Shape Building Escape\n(APF Wall-Following Protocol Activated)", fontsize=14, fontweight='bold')
    plt.xlabel("X (m)", fontsize=12)
    plt.ylabel("Y (m)", fontsize=12)
    plt.legend(loc='lower left')
    plt.axis([-2, 22, -8, 8])
    
    print("✅ [Layer 3 Visualizer] 模擬完成，已生成圖形。")
    plt.savefig("l3_ushape_escape.png")
    print("已儲存為 l3_ushape_escape.png")
    plt.show()

if __name__ == "__main__":
    plot_ushape_escape()
