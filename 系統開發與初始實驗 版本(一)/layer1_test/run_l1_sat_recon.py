import sys
import os
import json

# Ensure layer1 is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from layer1_standalone import Layer1Brain

def main():
    print("🧠 [Layer 1] 啟動真實 LLM 推論測試 (針對 Phase 3 軍事車輛偵蒐)")
    
    # 初始化 Layer 1 Brain (將使用 Ollama 或 Fallback)
    brain = Layer1Brain(db_dir="db")
    
    instruction = "在目前長寬為 600m x 450m 的衛星圖資(Global_Area)中，幫我搜索並標記所有潛在的軍事車輛 (包含裝甲車、坦克)。"
    
    current_state = {
        "battery": 100.0,
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "status": "IDLE"
    }
    
    print(f"\n🗣️ User Prompt: {instruction}")
    print("⏳ 等待 LLM 檢索 RAG 知識庫並進行推論...\n")
    
    try:
        decision = brain.decide(
            task_instruction=instruction,
            current_state=current_state,
            target_area="Global_Area",
            generate_target=True
        )
        
        print("\n✅ [LLM 最終輸出決策 (JSON)] ===")
        print(json.dumps(decision, indent=2, ensure_ascii=False))
        
        # 嘗試讀取生成的 Target Schema
        target_db_path = os.path.join("db", "targets_db.json")
        if os.path.exists(target_db_path):
            with open(target_db_path, "r", encoding="utf-8") as f:
                targets = json.load(f) # This is a list of dicts
                
            # 找尋剛生成的 target
            military_targets = [t for t in targets if "military" in t.get("chunk", "").lower() or "tank" in t.get("chunk", "").lower() or "vehicle" in t.get("chunk", "").lower() or "軍車" in t.get("chunk", "") or "軍事" in t.get("chunk", "")]
            if military_targets:
                latest_target = military_targets[-1]
                print(f"\n🎯 [動態生成的 Target Schema ({latest_target.get('id', 'Unknown')})] ===")
                print(json.dumps(latest_target, indent=2, ensure_ascii=False))
            else:
                print("\n🎯 [動態生成的 Target Schema] ===\n(未尋找到特定名稱，印出最後一個目標)")
                if targets:
                     print(json.dumps(targets[-1], indent=2, ensure_ascii=False))
                
    except Exception as e:
        print(f"\n❌ 推論失敗: {e}")

if __name__ == "__main__":
    main()
