"""
ollama_demo.py - Layer 1 真實 Ollama 推論測試
============================================
用於驗證 llava:13b 在 6 步驟管道下的 JSON 輸出品質。
"""
from layer1_standalone import Layer1Brain
import json
import time

def run_test():
    print("="*60)
    print("🚀 啟動 Layer 1 Brain (LLM: Ollama + llava:13b)")
    print("="*60)

    # 初始化腦部，使用 Ollama
    # 注意：第一次執行會由 SentenceTransformer 下載模型，請耐心等候
    brain = Layer1Brain(
        use_ollama=True,
        ollama_model="llava:13b",
        verbose=True
    )

    test_instructions = [
        "前往 Area1 森林區執行巡邏，注意高度限制",
        "在 Area3 搜索紅色可疑車輛，可能帶有武器",
        "通訊中斷，立即返回基地 RTH"
    ]

    for i, inst in enumerate(test_instructions):
        print(f"\n\n[測試案例 {i+1}] 指令: {inst}")
        start_time = time.time()
        
        # 模擬一些當前狀態
        state = {"Zone": "RT_FOR_01", "Battery": 85.0} if "Area1" in inst else {"Zone": "RT_OPE_01"}
        
        try:
            decision = brain.decide(
                task_instruction=inst,
                current_state=state,
                target_area="Area1" if "Area1" in inst else "Area3"
            )
            
            elapsed = time.time() - start_time
            print(f"\n✅ 推論完成，耗時: {elapsed:.2f}秒")
            print(f"指令結果: {decision.get('command')}")
            print(f"決策分析: {decision.get('reasoning')}")
            print(f"航點數量: {len(decision.get('waypoints', []))}")
            
        except Exception as e:
            print(f"❌ 測試失敗: {e}")

if __name__ == "__main__":
    run_test()
