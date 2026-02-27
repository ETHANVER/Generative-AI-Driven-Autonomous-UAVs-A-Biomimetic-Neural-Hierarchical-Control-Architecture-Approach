"""
mock_rag.py - Layer 1 RAG 引擎（不依賴 ChromaDB / OpenAI Embeddings）
======================================================================
根據論文第 4.1 節「RAG 向量檢索機制」設計：

  (1) 語意分塊與向量化 (Semantic Chunking & Embedding)
      → 以 TF-IDF 餘弦相似度模擬 text-embedding-3-small

  (2) 情境感知檢索 (Context-Aware Retrieval)
      → 將「當前狀態」結合「任務指令」生成混合查詢向量

  (3) Top-K 過濾 (K=3，相似度閾值 0.75)
      → 過濾語意雜訊

  (4) 動態 Prompt 注入 (Dynamic Prompt Injection)
      → 將過濾後的規則填入 [Constraints] 區塊

API 完全相容正式 ChromaDB 版本，切換只需更改 import。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    """基礎中英文斷詞：小寫、移除標點"""
    text = text.lower()
    # 保留中文字元、英文字母、數字
    tokens = re.findall(r'[\u4e00-\u9fff]|[a-z0-9]+', text)
    return tokens


def _tf(tokens: List[str]) -> Dict[str, float]:
    """計算詞頻 (Term Frequency)"""
    freq: Dict[str, float] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0.0) + 1.0
    n = len(tokens) or 1
    return {k: v / n for k, v in freq.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """計算稀疏向量的餘弦相似度"""
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ══════════════════════════════════════════════════════════════════════
# 向量儲存庫 (模擬 ChromaDB Collection)
# ══════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    輕量 TF-IDF 向量儲存庫。
    每個條目對應論文的一個「語意分塊 (Semantic Chunk)」。
    """

    def __init__(self, name: str):
        self.name = name
        self._docs: List[dict] = []         # 原始條目
        self._vecs: List[Dict[str, float]] = []  # TF-IDF 向量

    def add_documents(self, docs: List[dict], text_field: str = "chunk"):
        """批量加入文件，自動向量化"""
        for doc in docs:
            text = doc.get(text_field, "")
            vec = _tf(_tokenize(text))
            self._docs.append(doc)
            self._vecs.append(vec)
        print(f"  [VectorStore:{self.name}] 已載入 {len(docs)} 個語意分塊")

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        threshold: float = 0.75,
    ) -> List[Tuple[dict, float]]:
        """
        語意搜索。回傳 [(doc, similarity)] 列表，
        已依相似度降序排列並套用 threshold 過濾。
        """
        q_vec = _tf(_tokenize(query_text))
        scored = [
            (doc, _cosine(q_vec, vec))
            for doc, vec in zip(self._docs, self._vecs)
        ]
        # 依相似度降序排序
        scored.sort(key=lambda x: x[1], reverse=True)
        # Top-K 過濾 + 相似度閾值
        results = [(doc, sim) for doc, sim in scored[:top_k] if sim >= threshold]
        return results

    def get_all(self) -> List[dict]:
        return list(self._docs)

    def __len__(self):
        return len(self._docs)


# ══════════════════════════════════════════════════════════════════════
# 四大 RAG 資料庫
# ══════════════════════════════════════════════════════════════════════

class RAGDatabases:
    """
    論文第 4.1 節定義的四大 RAG 資料庫整合管理器。
    自動從 db/ 子目錄讀取 JSON 並建立向量索引。
    """

    DB_FILES = {
        "environment": "environment_db.json",
        "rules":       "rules_db.json",
        "conditions":  "conditions_db.json",
        "targets":     "targets_db.json",
    }

    def __init__(self, db_dir: str = "db"):
        self.db_dir = Path(db_dir)
        self.stores: Dict[str, VectorStore] = {}
        self._load_all()

    def _load_all(self):
        print("[RAGDatabases] 載入四大 RAG 資料庫...")
        for name, filename in self.DB_FILES.items():
            path = self.db_dir / filename
            try:
                with open(path, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                store = VectorStore(name)
                store.add_documents(docs, text_field="chunk")
                self.stores[name] = store
            except FileNotFoundError:
                print(f"  ⚠️  找不到資料庫檔案: {path}")
                self.stores[name] = VectorStore(name)
        print(f"[RAGDatabases] 四大資料庫已就緒：共 {self.total_chunks()} 個語意分塊\n")

    def total_chunks(self) -> int:
        return sum(len(s) for s in self.stores.values())

    def query_db(
        self,
        db_name: str,
        query_text: str,
        top_k: int = 3,
        threshold: float = 0.75,
    ) -> List[Tuple[dict, float]]:
        """查詢單一資料庫"""
        store = self.stores.get(db_name)
        if store is None:
            return []
        return store.query(query_text, top_k=top_k, threshold=threshold)


# ══════════════════════════════════════════════════════════════════════
# RAG 引擎（完整管道）
# ══════════════════════════════════════════════════════════════════════

class RAGEngine:
    """
    論文 4.1 節「RAG Implementation Pipeline」完整實作：

    步驟：
      1. 情境感知檢索 — 混合「當前狀態」+「任務指令」生成查詢
      2. 對四大資料庫分別執行 Top-K 語意搜索
      3. 相似度閾值 0.75 過濾
      4. 動態 Prompt 注入 — 輸出 [Constraints] 區塊文字
    """

    TOP_K = 3
    SIMILARITY_THRESHOLD = 0.75

    def __init__(self, databases: Optional[RAGDatabases] = None, db_dir: str = "db"):
        self.dbs = databases or RAGDatabases(db_dir)

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        task_instruction: str,
        current_state: Optional[dict] = None,
        target_area: str = "",
    ) -> Dict[str, List[Tuple[dict, float]]]:
        """
        情境感知檢索 (Context-Aware Retrieval)。

        Args:
            task_instruction: Layer 1 收到的原始任務指令（自然語言）
            current_state:    Layer 2 回報的當前狀態字典，如 {"Zone":"Area1","Event":"Target_Found"}
            target_area:      目標區域名稱，用於增強查詢精確度

        Returns:
            {db_name: [(doc, similarity), ...]} 字典
        """
        # ── 混合查詢向量生成 ──────────────────────────────
        state_text = ""
        if current_state:
            state_text = " ".join(f"{k} {v}" for k, v in current_state.items())
        combined_query = f"{task_instruction} {state_text} {target_area}".strip()

        print(f"  [RAGEngine] 混合查詢: «{combined_query[:80]}»")

        # ── 對四大資料庫分別檢索 ──────────────────────────
        results: Dict[str, List] = {}
        for db_name in self.dbs.stores:
            hits = self.dbs.query_db(
                db_name,
                combined_query,
                top_k=self.TOP_K,
                threshold=self.SIMILARITY_THRESHOLD,
            )
            results[db_name] = hits
            if hits:
                print(f"    📚 [{db_name}] 命中 {len(hits)} 條 "
                      f"(最高相似度: {hits[0][1]:.3f})")
            else:
                print(f"    📚 [{db_name}] 無命中（閾值 {self.SIMILARITY_THRESHOLD}）")

        return results

    def build_constraints_block(
        self,
        retrieval_results: Dict[str, List[Tuple[dict, float]]],
    ) -> str:
        """
        動態 Prompt 注入 (Dynamic Prompt Injection)：
        將檢索到的規則/條件/環境資訊組裝為 [Constraints] 區塊。
        """
        lines = ["[Constraints]"]

        for db_name, hits in retrieval_results.items():
            if not hits:
                continue
            label_map = {
                "environment": "🗺️ 環境",
                "rules":       "📋 規則",
                "conditions":  "⚙️ 條件",
                "targets":     "🎯 目標",
            }
            label = label_map.get(db_name, db_name)
            for doc, sim in hits:
                chunk = doc.get("chunk", "")
                lines.append(f"  • [{label}] (相似度:{sim:.2f}) {chunk}")

        if len(lines) == 1:
            lines.append("  （無相關約束條件，按通用 SOP 執行）")

        return "\n".join(lines)

    def full_pipeline(
        self,
        task_instruction: str,
        current_state: Optional[dict] = None,
        target_area: str = "",
    ) -> Tuple[Dict, str]:
        """
        執行完整 RAG 管道。
        回傳 (retrieval_results, constraints_block_text)
        """
        results = self.retrieve(task_instruction, current_state, target_area)
        constraints = self.build_constraints_block(results)
        return results, constraints

    def get_db_summary(self) -> str:
        """資料庫狀態摘要"""
        lines = ["=== RAG 資料庫狀態 ==="]
        for name, store in self.dbs.stores.items():
            lines.append(f"  {name:15s}: {len(store):3d} 個語意分塊")
        lines.append(f"  {'合計':15s}: {self.dbs.total_chunks():3d} 個")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 快速測試
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = RAGEngine(db_dir="db")
    print(engine.get_db_summary())

    print("\n" + "─" * 60)
    print("測試：森林巡邏任務 + Target Found 狀態")
    print("─" * 60)
    _, constraints = engine.full_pipeline(
        task_instruction="前往 Area1 森林區執行巡邏偵蒐任務",
        current_state={"Zone": "Area1", "Event": "Obstacle"},
        target_area="Area1",
    )
    print("\n" + constraints)
