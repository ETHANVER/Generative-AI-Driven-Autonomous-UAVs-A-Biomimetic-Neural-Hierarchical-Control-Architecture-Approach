"""
episodic_memory.py - Layer 1 側情節記憶 RAG 介面
==================================================
根據論文第 4.4 節「閉環學習機制」實作 Layer 1 側的情節記憶讀取：

  寫入 (Encoding)：由 Layer 2 完成（mock_memory.py / memory_db.py）
  反饋 (Retrieval)：本模組負責 — Layer 1 在生成決策前讀取歷史經驗，
                    將相關負面事件注入 System Prompt 的 [History] 區塊，
                    避免 Layer 1 重複犯下相同錯誤。

兩種運作模式（自動偵測）：
  1. 獨立模式：使用本模組內建的輕量記憶儲存（InMemoryStore）
  2. 橋接模式：從 layer2_test/mock_memory.py 橋接，共享 Layer 2 寫入的事件

介面設計：
  em = EpisodicMemory()
  em.log_event(zone, event_type, decision, outcome, note)     # 寫入
  snippets = em.retrieve_experience(query, top_k=3)           # 讀取
  block = em.build_history_block(query)                       # 生成 [History] 區塊
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════════
# 內建輕量記憶儲存（不依賴 ChromaDB）
# ══════════════════════════════════════════════════════════════════════

class InMemoryStore:
    """
    輕量 TF-IDF 記憶儲存，供 Layer 1 情節記憶使用。
    設計為持久化（寫入 episodic_memory.json），以跨 session 保留記憶。
    """

    def __init__(self, persist_path: str = "episodic_memory.json"):
        self.persist_path = Path(persist_path)
        self._entries: List[dict] = []
        self._load()

    def _load(self):
        if self.persist_path.exists():
            try:
                self._entries = json.loads(
                    self.persist_path.read_text(encoding="utf-8")
                )
                print(f"  [EpisodicMemory] 載入 {len(self._entries)} 條歷史記憶")
            except Exception:
                self._entries = []

    def _save(self):
        self.persist_path.write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def add(self, entry: dict):
        entry.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._entries.append(entry)
        self._save()

    def get_all(self) -> List[dict]:
        return list(self._entries)

    def __len__(self):
        return len(self._entries)


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r'[\u4e00-\u9fff]|[a-z0-9]+', text)


def _tf(tokens: List[str]) -> dict:
    freq: dict = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    n = len(tokens) or 1
    return {k: v / n for k, v in freq.items()}


def _cosine(a: dict, b: dict) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v*v for v in a.values()))
    mag_b = math.sqrt(sum(v*v for v in b.values()))
    return dot / (mag_a * mag_b + 1e-10)


# ══════════════════════════════════════════════════════════════════════
# 情節記憶主類別
# ══════════════════════════════════════════════════════════════════════

class EpisodicMemory:
    """
    Layer 1 情節記憶 RAG 介面。
    實現論文 4.4 節「閉環學習機制」的 Retrieval 階段：
      Layer 1 在決策前讀取歷史負面事件，避免重複犯錯。
    """

    NEGATIVE_LABELS = {
        "Geofencing_Blocked", "LLM_Hallucination", "Timeout",
        "CommLoss", "ObstacleFail", "BatteryEmergency", "ParseError"
    }
    POSITIVE_LABELS = {
        "Target_Found", "Mission_Complete", "Waypoint_Arrived"
    }

    def __init__(
        self,
        persist_path: str = "episodic_memory.json",
        bridge_path: Optional[str] = None,
    ):
        """
        Args:
            persist_path:  本地記憶持久化 JSON 路徑
            bridge_path:   Layer 2 mock_memory 的 JSON 路徑（橋接共享記憶）
        """
        self._store = InMemoryStore(persist_path)
        self._bridge_path = Path(bridge_path) if bridge_path else None

        # 嘗試橋接 Layer 2 記憶
        self._bridge_entries: List[dict] = []
        self._load_bridge()

    def _load_bridge(self):
        """載入 Layer 2（mock_memory.py）寫入的情節記憶"""
        if self._bridge_path and self._bridge_path.exists():
            try:
                data = json.loads(self._bridge_path.read_text(encoding="utf-8"))
                self._bridge_entries = data if isinstance(data, list) else []
                print(f"  [EpisodicMemory] 橋接 Layer 2 記憶：{len(self._bridge_entries)} 條")
            except Exception as e:
                print(f"  [EpisodicMemory] 橋接失敗: {e}")

    # ------------------------------------------------------------------
    # 公開 API — 寫入
    # ------------------------------------------------------------------

    def log_event(
        self,
        zone: str,
        event_type: str,
        decision: str,
        outcome: str,
        note: str = "",
        label: str = "Neutral",
    ):
        """
        寫入情節記憶條目。
        Layer 1 自身也可以記錄決策結果，供下次決策參考。

        Args:
            zone:       區域名稱（如 "RT_FOR_01", "Area2"）
            event_type: 事件類型（如 "LLM_Hallucination", "Target_Found"）
            decision:   Layer 1 做出的決策（如 "PATROL Area1"）
            outcome:    結果（如 "Blocked by Geofencing", "Success"）
            note:       補充說明
            label:      標籤："Negative" | "Positive" | "Neutral"
        """
        entry = {
            "zone":       zone,
            "event_type": event_type,
            "decision":   decision,
            "outcome":    outcome,
            "note":       note,
            "label":      label,
            "text":       f"{zone} {event_type} {decision} {outcome} {note}",
        }
        self._store.add(entry)

    # ------------------------------------------------------------------
    # 公開 API — 檢索
    # ------------------------------------------------------------------

    def retrieve_experience(
        self,
        query: str,
        top_k: int = 3,
        min_sim: float = 0.15,
        prefer_negative: bool = True,
    ) -> List[dict]:
        """
        論文 4.4 節 Retrieval 階段：
        根據當前任務查詢語意相似的歷史經驗。

        Args:
            query:           查詢字串（混合「當前狀態」+「任務指令」）
            top_k:           回傳最多 k 條記憶
            min_sim:         最低相似度閾值
            prefer_negative: 優先回傳負面記憶（避免重蹈覆轍）

        Returns:
            相似的情節記憶條目列表
        """
        all_entries = self._store.get_all() + self._bridge_entries
        if not all_entries:
            return []

        q_vec = _tf(_tokenize(query))
        scored = []
        for entry in all_entries:
            text = entry.get("text", " ".join(
                str(v) for v in entry.values() if isinstance(v, str)
            ))
            sim = _cosine(q_vec, _tf(_tokenize(text)))
            scored.append((entry, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 優先負面記憶（閉環學習核心）
        if prefer_negative:
            negatives = [
                (e, s) for e, s in scored
                if e.get("label") == "Negative" or
                   e.get("event_type") in self.NEGATIVE_LABELS
            ]
            others = [
                (e, s) for e, s in scored
                if e not in [x[0] for x in negatives]
            ]
            ordered = negatives + others
        else:
            ordered = scored

        results = [
            e for e, s in ordered[:top_k]
            if s >= min_sim
        ]
        return results

    def build_history_block(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        """
        生成 System Prompt 的 [History] 區塊（論文動態 Prompt 注入）。
        將歷史負面事件注入 Prompt，使 LLM 了解過去失敗案例。
        """
        memories = self.retrieve_experience(query, top_k=top_k)
        if not memories:
            return "[History]\n  （暫無相關歷史記憶）"

        lines = ["[History] — 過去的任務經驗（請避免重複失敗）"]
        for mem in memories:
            label   = mem.get("label", "Neutral")
            icon    = "⚠️" if label == "Negative" else "✅" if label == "Positive" else "ℹ️"
            zone    = mem.get("zone", "")
            evtype  = mem.get("event_type", "")
            outcome = mem.get("outcome", "")
            note    = mem.get("note", "")
            ts      = mem.get("timestamp", "")[:10]
            lines.append(
                f"  {icon} [{ts}] 區域:{zone} | 事件:{evtype} | 結果:{outcome}"
                + (f" | {note}" if note else "")
            )

        return "\n".join(lines)

    def summary(self) -> str:
        """統計摘要"""
        entries = self._store.get_all() + self._bridge_entries
        neg_cnt = sum(
            1 for e in entries
            if e.get("label") == "Negative" or e.get("event_type") in self.NEGATIVE_LABELS
        )
        return (
            f"情節記憶：{len(entries)} 條（本地 {len(self._store)} + "
            f"Layer2橋接 {len(self._bridge_entries)}），負面事件 {neg_cnt} 條"
        )


# ══════════════════════════════════════════════════════════════════════
# 預置測試記憶（讓 Layer 1 一開始就有歷史可參考）
# ══════════════════════════════════════════════════════════════════════

def seed_test_memories(mem: EpisodicMemory):
    """
    植入模擬歷史記憶（用於測試與 Demo）。
    模擬 Layer 2 在過去任務中記錄的關鍵負面事件。
    """
    test_events = [
        {
            "zone": "RT_BUI_01", "event_type": "Geofencing_Blocked",
            "decision": "PATROL RT_BUI_01 z=-20m",
            "outcome": "Layer 2 攔截：建築禁航區低空飛行",
            "note": "Layer 1 未遵守高度限制（應 z < -50m）",
            "label": "Negative",
        },
        {
            "zone": "RT_FOR_01", "event_type": "CommLoss",
            "decision": "RECON Area1",
            "outcome": "通訊中斷 620ms，切換自主模式",
            "note": "森林區 RF 信號衰減，建議降低飛行速度",
            "label": "Negative",
        },
        {
            "zone": "RT_OPE_01", "event_type": "Target_Found",
            "decision": "SEARCH Open_Field",
            "outcome": "發現可疑人員，YOLO confidence=0.91",
            "note": "低空 8m 搜索效果最佳",
            "label": "Positive",
        },
        {
            "zone": "RT_BUI_01", "event_type": "LLM_Hallucination",
            "decision": "PATROL Building_Zone z=-30m",
            "outcome": "Layer 2 Geofencing 攔截（第 2 次）",
            "note": "Layer 1 對禁航區限制遺忘，需加強 [Constraints] 注入",
            "label": "Negative",
        },
        {
            "zone": "RT_FOR_01", "event_type": "Timeout",
            "decision": "RECON Forest 航點序列",
            "outcome": "任務逾時（90s），未完成全部航點",
            "note": "林區航點數過多，建議減至 4 個以內",
            "label": "Negative",
        },
    ]
    for ev in test_events:
        mem.log_event(**ev)
    print(f"  [EpisodicMemory] 已植入 {len(test_events)} 條測試歷史記憶")


if __name__ == "__main__":
    mem = EpisodicMemory(persist_path="episodic_memory_test.json")
    seed_test_memories(mem)
    print(mem.summary())

    query = "前往森林區執行偵蒐任務 建築"
    block = mem.build_history_block(query)
    print(f"\n生成的 [History] 區塊：\n{block}")
