"""
mock_memory.py - 模擬 EpisodicMemory（不依賴 ChromaDB）
=========================================================
用途：在獨立測試環境中取代正式的 memory_db.EpisodicMemory，
      避免安裝 chromadb 向量資料庫的依賴。
      API 完全相容正式版，切換方式只需更改 import。
"""
import uuid
import datetime
from typing import List, Optional


class EpisodicMemory:
    """
    輕量模擬版情節記憶 (Episodic Memory)
    以純 Python list 作為儲存後端，完整模擬正式版行為。
    """

    def __init__(self, db_path: str = "./mock_memory_data"):
        """
        db_path 僅保留以維持 API 相容性（本版本不使用磁碟）
        """
        self._events: list = []
        print(f"[MockMemory] 初始化完成 (In-Memory mode, path 參數被忽略: {db_path})")

    # ------------------------------------------------------------------
    # Encoding 階段：寫入事件
    # ------------------------------------------------------------------
    def write_event(
        self,
        event_type: str,
        description: str,
        severity: str = "INFO",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        記錄關鍵事件。
        回傳事件 ID（uuid 字串）。
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        is_negative = severity in ("WARNING", "ERROR")

        record = {
            "id": event_id,
            "timestamp": timestamp,
            "type": event_type,
            "description": description,
            "severity": severity,
            "is_negative": is_negative,
        }
        if metadata:
            record.update(metadata)

        self._events.append(record)

        flag = "⚠️ 異常" if is_negative else "✅ 正常"
        print(
            f"🎬 [情節記憶] 寫入 | {flag} | 類型:{event_type} | {description}"
        )
        return event_id

    # ------------------------------------------------------------------
    # Retrieval 階段：簡易關鍵字過濾檢索
    # ------------------------------------------------------------------
    def retrieve_lessons_learned(
        self, current_context: str, top_k: int = 3
    ) -> List[str]:
        """
        依關鍵字相關性檢索過往「負面」事件，模擬 ChromaDB 語意搜索。
        回傳最多 top_k 筆描述字串。
        """
        negative_events = [e for e in self._events if e["is_negative"]]

        # 簡易相關性評分：計算 context 中的詞彙出現次數
        def relevance(event: dict) -> int:
            text = f"{event['type']} {event['description']}".lower()
            return sum(1 for word in current_context.lower().split() if word in text)

        ranked = sorted(negative_events, key=relevance, reverse=True)[:top_k]

        if not ranked:
            return ["無過往異常紀錄，可按標準 SOP 執行。"]

        return [
            f"[{e['timestamp'][:19]}] 類型:{e['type']} | {e['description']} | 嚴重度:{e['severity']}"
            for e in ranked
        ]

    # ------------------------------------------------------------------
    # 輔助方法
    # ------------------------------------------------------------------
    def get_summary_report(self) -> str:
        """生成任務簡報"""
        total = len(self._events)
        neg = sum(1 for e in self._events if e["is_negative"])
        if total == 0:
            return "目前無記憶資料。"
        return (
            f"總計記錄事件: {total} 筆\n"
            f"負面異常回饋: {neg} 筆\n"
            f"正常事件: {total - neg} 筆"
        )

    def get_all_events(self) -> list:
        """回傳所有事件（測試用）"""
        return list(self._events)

    def clear(self):
        """清空所有記憶（測試用）"""
        self._events.clear()
        print("[MockMemory] 記憶已清空")


# ------------------------------------------------------------------
# 快速測試
# ------------------------------------------------------------------
if __name__ == "__main__":
    mem = EpisodicMemory()

    mem.write_event("GEOFENCE_VIOLATION", "飛入 Area2 紅色禁航區", severity="WARNING")
    mem.write_event("TARGET_FOUND", "偵測到可疑車輛於 (50, 30)", severity="INFO")
    mem.write_event("TIMEOUT", "Layer1 API 逾時超過 500ms", severity="ERROR")

    print("\n--- 檢索與 Area2 相關的負面經驗 ---")
    lessons = mem.retrieve_lessons_learned("Area2 禁航區邊界")
    for lesson in lessons:
        print(f"  🧠 {lesson}")

    print("\n--- 任務簡報 ---")
    print(mem.get_summary_report())
