"""
run_demo.py - Layer 2 Behavior Planner 互動式 Demo
====================================================
展示論文「基於仿生神經分工架構之生成式 AI 自主無人機研究」
第 4.2 節的完整 Layer 2 運作流程。

不需要 AirSim / OpenAI / ChromaDB，全程在純 Python 環境執行。

使用方式：
  python run_demo.py
"""

import time
import sys

from mock_memory import EpisodicMemory
from layer2_standalone import (
    FSM, FSMState,
    Waypoint,
    Layer2BehaviorPlanner,
)

DIVIDER = "─" * 62


def header(title: str):
    print(f"\n{'═' * 62}")
    print(f"  {title}")
    print(f"{'═' * 62}")


def step(num: int, desc: str):
    print(f"\n{'─' * 62}")
    print(f"  步驟 {num}：{desc}")
    print(f"{'─' * 62}")


def pause(sec: float = 0.8):
    time.sleep(sec)


# ══════════════════════════════════════════════════════════════════════
# Demo 場景定義
# ══════════════════════════════════════════════════════════════════════

# ── 場景 A：正常巡邏任務 ──────────────────────────────────────────
SCENARIO_A = {
    "mission_id": "DEMO_PATROL_001",
    "command": "PATROL",
    "target_area": "Area1",
    "waypoints": [
        {"x": 10, "y": 10, "z": -10, "label": "起點_Takeoff"},
        {"x": 30, "y": 20, "z": -15, "label": "巡邏點_Alpha"},
        {"x": 60, "y": 50, "z": -15, "label": "巡邏點_Beta"},
        {"x": 80, "y": 70, "z": -10, "label": "巡邏點_Gamma"},
        {"x": 50, "y": 50, "z": -10, "label": "返航待命點"},
    ],
    "altitude": -10.0,
    "speed": 5.0,
    "priority": 7,
    "timeout_sec": 30.0,
}

# ── 場景 B：含禁航區的非法任務（模擬 LLM 幻覺） ────────────────
SCENARIO_B = {
    "mission_id": "DEMO_HALLUCINATION_002",
    "command": "RECON",
    "target_area": "Area2",
    "waypoints": [
        {"x": 50, "y": 50, "label": "合法點"},
        {"x": 150, "y": 150, "label": "非法點_禁航區"},   # Area2 高危
        {"x": 80, "y": 30, "label": "另一合法點"},
    ],
    "altitude": -12.0,
    "speed": 3.0,
    "priority": 8,
    "timeout_sec": 20.0,
}

# ══════════════════════════════════════════════════════════════════════
# 主 Demo 流程
# ══════════════════════════════════════════════════════════════════════

def run_demo():
    header("Layer 2 Behavior Planner — 互動式 Demo")
    print("  論文：基於仿生神經分工架構之生成式 AI 自主無人機研究")
    print("  模式：純模擬（無 AirSim / OpenAI / ChromaDB 依賴）")
    print()
    print("  架構重點 (論文 4.2 節)：")
    print("  • FSM 有限狀態機    → 防止競態條件，管理任務生命週期")
    print("  • Geofencing        → 攔截 LLM 幻覺產生的非法座標")
    print("  • Code-as-Planner   → 抽象決策轉譯為具體航點序列")
    print("  • 10Hz 監控迴圈     → 即時追蹤航點抵達狀態")
    print("  • 斷線自主模式      → 情節記憶庫自主導航後備")

    input("\n  按 Enter 開始 Demo...")

    # ── 初始化 Layer2 ──────────────────────────────────────────────
    memory = EpisodicMemory()
    # 預置一些歷史記憶（模擬閉環學習）
    memory.write_event("GEOFENCE_VIOLATION", "舊任務曾嘗試進入 Area2 禁航區", severity="WARNING")
    memory.write_event("PATROL_SUCCESS", "Area1 森林區巡邏完成，無異常", severity="INFO")

    layer2 = Layer2BehaviorPlanner(env_path="environment.json", memory=memory)

    # ══════════════════════════════════════════════════════════════
    # 場景 A：正常任務流程
    # ══════════════════════════════════════════════════════════════
    header("場景 A：正常巡邏任務 (PATROL)")

    step(1, "Layer 1 (LLM) 輸出 JSON 決策")
    print("  模擬 GPT-4o 輸出的任務決策：")
    print(f"  任務 ID : {SCENARIO_A['mission_id']}")
    print(f"  指令   : {SCENARIO_A['command']}")
    print(f"  目標區 : {SCENARIO_A['target_area']} (森林巡邏)")
    print(f"  航點數 : {len(SCENARIO_A['waypoints'])} 個")
    pause()

    step(2, "Layer 2 接收並執行 Geofencing 驗證")
    success = layer2.process_layer1_decision(SCENARIO_A)

    if success:
        print("\n  ✅ 任務通過驗證，開始執行！")
        print(f"  📍 FSM 狀態: {layer2.fsm.state.name}")
    else:
        print("  ❌ 任務被攔截（不應發生於場景 A）")
    pause()

    step(3, "模擬無人機飛行（逐點抵達）")
    waypoints_to_visit = [
        (10, 10, -10),
        (30, 20, -15),
        (60, 50, -15),
        (80, 70, -10),
        (50, 50, -10),
    ]
    for i, (x, y, z) in enumerate(waypoints_to_visit):
        layer2.simulate_drone_move(x, y, z)
        print(f"  🚁 無人機移動至航點 [{i}] → ({x}, {y}, {z})")
        time.sleep(0.3)
        if layer2.fsm.state == FSMState.MISSION_DONE:
            print("\n  🏁 任務完成！")
            break

    # 等待監控迴圈確認完成
    timeout = time.time() + 5
    while layer2.fsm.state == FSMState.EXECUTING and time.time() < timeout:
        time.sleep(0.1)

    print(f"\n  最終 FSM 狀態: {layer2.fsm.state.name}")
    pause()

    step(4, "情節記憶報告")
    print(memory.get_summary_report())

    layer2.stop()
    FSM().reset()
    input("\n  場景 A 完成。按 Enter 繼續場景 B...")

    # ══════════════════════════════════════════════════════════════
    # 場景 B：LLM 幻覺攔截
    # ══════════════════════════════════════════════════════════════
    header("場景 B：Geofencing 攔截 LLM 幻覺")

    memory2 = EpisodicMemory()
    memory2.write_event("PATROL", "Area1 正常巡邏", severity="INFO")
    layer2b = Layer2BehaviorPlanner(env_path="environment.json", memory=memory2)

    step(5, "Layer 1 輸出含禁航區座標的非法決策（模擬 LLM 幻覺）")
    print("  ⚠️  LLM 生成了包含禁航區座標的任務！")
    print(f"  任務 ID : {SCENARIO_B['mission_id']}")
    for wp in SCENARIO_B['waypoints']:
        print(f"          航點: ({wp['x']}, {wp['y']}) — {wp['label']}")
    pause()

    step(6, "Layer 2 Geofencing 攔截")
    success = layer2b.process_layer1_decision(SCENARIO_B)

    if not success:
        print(f"\n  ✅ 幻覺攔截成功！FSM 狀態: {layer2b.fsm.state.name}")
        print("  ✅ 非法任務已被 Geofencing 機制阻止，寫入情節記憶。")
    else:
        print("  ❌ 幻覺未被攔截（測試失敗）")

    pause()

    step(7, "查詢情節記憶中的負面經驗")
    lessons = memory2.retrieve_lessons_learned("Area2 禁航區 GEOFENCE_VIOLATION")
    print("  🧠 Layer 1 在下次決策前將見到這些教訓：")
    for lesson in lessons:
        print(f"     • {lesson}")

    layer2b.stop()
    FSM().reset()
    input("\n  場景 B 完成。按 Enter 繼續場景 C...")

    # ══════════════════════════════════════════════════════════════
    # 場景 C：斷線自主模式
    # ══════════════════════════════════════════════════════════════
    header("場景 C：通訊中斷 → 自主模式")

    memory3 = EpisodicMemory()
    memory3.write_event("PATROL", "Area1 歷史巡邏路線: (20,30)→(60,50)", severity="INFO")
    layer2c = Layer2BehaviorPlanner(env_path="environment.json", memory=memory3)

    step(8, "正常任務運行中...")
    layer2c.process_layer1_decision({
        "mission_id": "DEMO_AUTONOMOUS",
        "command": "PATROL",
        "target_area": "Area1",
        "waypoints": [{"x": 20, "y": 30}, {"x": 60, "y": 50}],
        "altitude": -10.0,
        "speed": 5.0,
        "priority": 5,
        "timeout_sec": 60.0,
    })
    print("  任務執行中...")
    pause(1.0)

    step(9, "模擬 Layer 1 API 通訊中斷（超過 500ms 逾時）")
    print("  ⚡ 模擬雲端 API 斷線...")
    layer2c.activate_autonomous_mode("Layer 1 API 逾時 (> 500ms)")

    print(f"\n  FSM 切換至: {layer2c.fsm.state.name}")
    print("  🤖 無人機正依據情節記憶庫中的歷史路線自主導航...")

    layer2c.stop()

    # ══════════════════════════════════════════════════════════════
    # 結語
    # ══════════════════════════════════════════════════════════════
    header("Demo 完成 — 總結")
    print("  Layer 2 Behavior Planner 三大核心機制驗證：")
    print()
    print("  ✅ 場景 A：FSM + 路徑規劃 + 10Hz 監控迴圈")
    print("             → 正常任務從啟動到完成的全生命週期管理")
    print()
    print("  ✅ 場景 B：Geofencing 地理圍欄攔截")
    print("             → 有效攔截 LLM 幻覺產生的禁航區指令")
    print("             → 異常事件自動寫入情節記憶，形成閉環學習")
    print()
    print("  ✅ 場景 C：斷線自主模式")
    print("             → API 中斷後無縫切換本地情節記憶導航")
    print()
    print("  下一步：接入 AirSim 替換 simulate_drone_move() 的模擬位置，")
    print("          並連接真實 Layer 1 LLM 輸出，即可完成 SITL 驗證。")
    print()


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n  [Demo 已被使用者中斷]")
        sys.exit(0)
