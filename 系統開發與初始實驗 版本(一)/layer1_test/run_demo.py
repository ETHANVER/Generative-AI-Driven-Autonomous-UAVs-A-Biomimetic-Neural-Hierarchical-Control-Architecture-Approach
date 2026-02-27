"""
run_demo.py - Layer 1 Brain 互動式 Demo
=========================================
展示論文第 4.1 節的完整 Layer 1 決策管道：

  場景 A：正常任務 — 自然語言指令 → RAG 檢索 → JSON 決策
  場景 B：RAG 效果對比 — 有/無 RAG Constraints 的差異
  場景 C：電量安全機制 — 低電量強制 RTH
  場景 D：閉環學習展示 — Layer 2 狀態回報觸發 RAG 調整

不需要 AirSim / OpenAI / ChromaDB，全程純 Python 執行。
"""
import json
import sys
import time

from mock_rag import RAGEngine
from mock_llm import MockLLM
from layer1_standalone import Layer1Brain


DIVIDER = "─" * 62


def header(title: str):
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


def step(num: int, desc: str):
    print(f"\n{DIVIDER}")
    print(f"  步驟 {num}：{desc}")
    print(DIVIDER)


def pause(sec: float = 0.6):
    time.sleep(sec)


def print_decision(decision: dict):
    """格式化輸出 Layer 1 決策"""
    print(f"\n  📋 Layer 1 JSON 決策輸出：")
    print(f"  ┌──────────────────────────────────────────────")
    print(f"  │ mission_id  : {decision.get('mission_id', 'N/A')}")
    print(f"  │ command     : {decision.get('command', 'N/A')}")
    print(f"  │ target_area : {decision.get('target_area', 'N/A')}")
    print(f"  │ waypoints   : {len(decision.get('waypoints', []))} 個")
    print(f"  │ altitude    : {decision.get('altitude', 0)} m (NED)")
    print(f"  │ speed       : {decision.get('speed', 0)} m/s")
    print(f"  │ timeout     : {decision.get('timeout_sec', 0)} s")
    print(f"  │ 推論時間    : {decision.get('_inference_ms', 0)} ms")
    rag_hits = decision.get('_rag_hit_counts', {})
    print(f"  │ RAG 命中    : {rag_hits}")
    print(f"  │ 推論說明    : {decision.get('reasoning', '')[:80]}")
    print(f"  └──────────────────────────────────────────────")

    wps = decision.get("waypoints", [])
    if wps:
        print(f"\n  ✈️  航點序列：")
        for i, wp in enumerate(wps):
            label = f" ({wp.get('label', '')})" if wp.get("label") else ""
            print(f"    [{i}] ({wp['x']}, {wp['y']}, {wp['z']}){label}")


# ══════════════════════════════════════════════════════════════════════
# 主 Demo
# ══════════════════════════════════════════════════════════════════════

def run_demo():
    header("Layer 1 Brain 互動式 Demo")
    print("  論文：基於仿生神經分工架構之生成式 AI 自主無人機研究")
    print("  模式：純模擬（無 AirSim / OpenAI / ChromaDB 依賴）")
    print()
    print("  四大 RAG 資料庫：")
    print("  • 環境資料庫 (Environment DB) — 語意網格地圖、區域邊界")
    print("  • 規則資料庫 (Rules DB)       — 飛行 SOP、安全規則")
    print("  • 條件資料庫 (Conditions DB)  — 硬體限制、觸發條件")
    print("  • 目標資料庫 (Target DB)      — 偵蒐目標特徵模板")
    print()
    print("  RAG 管道（論文 4.1 節 4 步驟）：")
    print("  ① 語意分塊向量化  →  ② 情境感知檢索")
    print("  ③ Top-K=3 閾值過濾 →  ④ 動態 Prompt 注入")

    input("\n  按 Enter 開始 Demo...")

    # ── 初始化 Layer 1 Brain──────────────────────────────
    brain = Layer1Brain(db_dir="db", verbose=False)
    print("\n" + brain.get_db_stats())

    # ══════════════════════════════════════════════════════
    # 場景 A：正常任務流程
    # ══════════════════════════════════════════════════════
    header("場景 A：完整 Layer 1 決策管道（森林搜索任務）")

    step(1, "使用者輸入自然語言任務指令")
    task = "請前往 Area1 森林區，執行偵蒐任務，尋找可疑人員，注意保持適當高度"
    print(f"  📝 任務指令: 「{task}」")
    pause()

    step(2, "RAG 管道執行（4 步驟）")
    print("  正在執行 RAG 四步驟管道...")
    brain.verbose = True
    rag_only = brain.explain_constraints(
        task_instruction=task,
        current_state={"Zone": "Area1", "Event": "None"},
    )
    print(f"\n  RAG 注入的 [Constraints] 區塊：\n{rag_only}")
    brain.verbose = False
    pause()

    step(3, "LLM 推論 → JSON 決策輸出（傳送至 Layer 2）")
    decision = brain.decide(
        task_instruction=task,
        current_state={"Zone": "Area1", "Event": "None"},
        target_area="Area1",
        battery_pct=85.0,
    )
    print_decision(decision)
    pause()

    input("\n  場景 A 完成。按 Enter 繼續場景 B...")

    # ══════════════════════════════════════════════════════
    # 場景 B：RAG 效果對比
    # ══════════════════════════════════════════════════════
    header("場景 B：RAG Constraints 效果展示")

    step(4, "比較：有/無 RAG 約束的 System Prompt 差異")

    # 無 RAG 的基礎 Prompt
    print("  ❌ 無 RAG 注入的基礎 System Prompt：")
    print("  ┌──────────────────────────────────────")
    print("  │ 你是無人機 AI，請生成任務 JSON。")
    print("  │ （無任何約束條件）")
    print("  └──────────────────────────────────────")
    pause(0.5)

    # 有 RAG 的完整 Prompt
    print("\n  ✅ 有 RAG 注入的增強 System Prompt：")
    engine = RAGEngine(db_dir="db")
    _, constraints = engine.full_pipeline(
        "前往 Area1 偵察目標",
        {"Zone": "Area1"},
        "Area1",
    )
    print(constraints)

    step(5, "展示不同任務觸發不同 RAG 約束組合")
    tests = [
        ("Area2 繞飛建築任務", {"Zone": "Area2"}, "Area2"),
        ("低空搜索開闊平原", {"Zone": "Area3"}, "Area3"),
        ("偵察人員目標", {"Event": "Target_Personnel"}, "Area1"),
    ]
    for instruction, state, area in tests:
        print(f"\n  查詢: 「{instruction}」")
        _, c = engine.full_pipeline(instruction, state, area)
        hits = sum(1 for line in c.split("\n") if "•" in line)
        print(f"  → RAG 命中 {hits} 條約束規則")
        pause(0.3)

    input("\n  場景 B 完成。按 Enter 繼續場景 C...")

    # ══════════════════════════════════════════════════════
    # 場景 C：電量安全機制
    # ══════════════════════════════════════════════════════
    header("場景 C：電量安全機制 — 強制 RTH")

    levels = [80.0, 35.0, 19.0]
    instructions = [
        "前往 Area1 森林偵察",
        "繼續前往 Area1 完成任務",
        "繼續執行任務！前往 Area3",
    ]
    for batt, task_inst in zip(levels, instructions):
        step_label = f"電量 {batt}%"
        print(f"\n  ⚡ {step_label}")
        print(f"     指令: 「{task_inst}」")

        decision = brain.decide(
            task_instruction=task_inst,
            battery_pct=batt,
            target_area="Area1",
        )
        cmd = decision.get("command")
        icon = "🔴 強制RTH" if cmd == "RTH" else "✅ 正常執行"
        print(f"     結果: {icon} → command={cmd} | waypoints={len(decision.get('waypoints', []))}")
        pause(0.5)

    input("\n  場景 C 完成。按 Enter 繼續場景 D...")

    # ══════════════════════════════════════════════════════
    # 場景 D：閉環學習 — Layer 2 狀態回報
    # ══════════════════════════════════════════════════════
    header("場景 D：閉環學習 — Layer 2 狀態回報影響 RAG 檢索")

    step(6, "Layer 2 接收 Target Found 訊號，回報至 Layer 1")
    layer2_state = {"Zone": "Area1", "Event": "Target_Found", "Confidence": "0.92"}
    print(f"  Layer 2 回報狀態: {layer2_state}")

    print("\n  Layer 1 根據新狀態重新執行 RAG 查詢...")
    _, constraints = engine.full_pipeline(
        task_instruction="目標發現，立即確認目標",
        current_state=layer2_state,
        target_area="Area1",
    )
    print(constraints)

    step(7, "Layer 1 生成新的「懸停偵查」決策")
    decision2 = brain.decide(
        task_instruction="目標發現，執行懸停偵查確認目標身份",
        current_state=layer2_state,
        target_area="Area1",
        battery_pct=70.0,
    )
    print_decision(decision2)

    # ══════════════════════════════════════════════════════
    # 結語
    # ══════════════════════════════════════════════════════
    header("Demo 完成 — 總結")
    print("  Layer 1 Brain 四大核心機制驗證：")
    print()
    print("  ✅ 場景 A：完整 4 步驟 RAG 管道")
    print("             語意分塊 → 情境感知 → Top-K 過濾 → Prompt 注入 → LLM 決策")
    print()
    print("  ✅ 場景 B：RAG 效果對比")
    print("             動態根據任務類型與目標區域注入不同約束組合")
    print()
    print("  ✅ 場景 C：電量安全機制")
    print("             電量 < 20% 強制 RTH，不可被任何指令覆蓋")
    print()
    print("  ✅ 場景 D：閉環學習（Layer 2 → Layer 1 回報）")
    print("             Layer 2 狀態回報觸發 RAG 重新查詢，動態調整決策")
    print()
    print("  下一步整合:")
    print("  • mock_llm.py → 改為真實 get_llm_client(use_openai=True, api_key='sk-...')")
    print("  • mock_rag.py → 改為正式 ChromaDB + text-embedding-3-small")
    print()


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n  [Demo 已被使用者中斷]")
        sys.exit(0)
