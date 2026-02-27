"""
rag_engine.py - Layer 1 RAG 引擎（論文符合版）
================================================
根據論文第 4.1 節「RAG Implementation Pipeline」：

  ✅ ChromaDB + HNSW（論文指定，免費開源）
  ✅ paraphrase-multilingual-MiniLM-L6-v2（免費替代 text-embedding-3-small）

四步驟管道：
  (1) Semantic Chunking & Embedding  — SentenceTransformer 向量化
  (2) Context-Aware Retrieval        — 混合狀態 + 任務指令查詢
  (3) Top-K=3 + 閾值 0.75 過濾
  (4) Dynamic Prompt Injection       — 輸出 [Constraints] 區塊

自動降級策略（確保測試不斷線）：
  - sentence-transformers 載入失敗 → 切換 TF-IDF 模式（維持 API 相容）
  - ChromaDB 操作失敗 → 切換到純記憶體模式
"""
from __future__ import annotations

import json
import math
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
# Embedding 後端（sentence-transformers or TF-IDF fallback）
# ══════════════════════════════════════════════════════════════════════

_EMBED_MODEL = None
_EMBED_MODE  = "loading"   # "sentence_transformers" | "tfidf"

def _load_embedding_model():
    global _EMBED_MODEL, _EMBED_MODE
    # 嘗試的模型列表
    models_to_try = [
        'paraphrase-multilingual-MiniLM-L6-v2',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2',
        'all-MiniLM-L6-v2'
    ]
    
    last_err = None
    from sentence_transformers import SentenceTransformer
    
    for model_name in models_to_try:
        try:
            print(f"  [Embedding] 嘗試載入 {model_name} ...")
            _EMBED_MODEL = SentenceTransformer(model_name)
            _EMBED_MODE  = "sentence_transformers"
            print(f"  [Embedding] ✅ {model_name} 就緒")
            return
        except Exception as e:
            last_err = e
            continue

    print(f"  [Embedding] ⚠️  所有模型均無法載入 ({last_err})，改用 TF-IDF fallback")
    _EMBED_MODE = "tfidf"


def _embed(texts: List[str]) -> List:
    """
    將文字列表轉換為向量（自動選擇後端）。
    回傳：sentence-transformers → numpy array 列表
          TF-IDF → dict 列表
    """
    global _EMBED_MODEL, _EMBED_MODE
    if _EMBED_MODE == "loading":
        _load_embedding_model()

    if _EMBED_MODE == "sentence_transformers" and _EMBED_MODEL is not None:
        vecs = _EMBED_MODEL.encode(texts, normalize_embeddings=True)
        return list(vecs)
    else:
        return [_tf(_tokenize(t)) for t in texts]


def _to_float_list(vec) -> List[float]:
    """
    將 numpy array 或任意可迭代物展間為純 Python float 列表。
    解决 ChromaDB 1.5.x 對 numpy float32 的格式訊求問題。
    """
    return [float(x) for x in vec]



def _cosine_dense(a, b) -> float:
    """Dense vector（numpy array）餘弦相似度"""
    import numpy as np
    return float(np.dot(a, b))   # 已 L2-normalize，dot product = cosine


def _tokenize(text: str) -> List[str]:
    return re.findall(r'[\u4e00-\u9fff]|[a-z0-9]+', text.lower())


def _tf(tokens: List[str]) -> dict:
    freq: dict = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    n = len(tokens) or 1
    return {k: v/n for k, v in freq.items()}


def _cosine_sparse(a: dict, b: dict) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot  = sum(a[k]*b[k] for k in common)
    ma   = math.sqrt(sum(v*v for v in a.values()))
    mb   = math.sqrt(sum(v*v for v in b.values()))
    return dot / (ma * mb + 1e-10)


# ══════════════════════════════════════════════════════════════════════
# VectorStore (ChromaDB Collection 封裝)
# ══════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    ChromaDB Collection 封裝。
    API 與舊 TF-IDF 版本完全相容（query / add_documents）。
    """

    def __init__(self, name: str, chroma_client=None, persist_dir: str = ".chroma_db"):
        self.name    = name
        self._docs:  List[dict] = []
        self._vecs   = []
        self._mode   = "memory"   # "chromadb" | "memory"
        self._collection = None

        # 嘗試初始化 ChromaDB
        if chroma_client is not None and _EMBED_MODE == "sentence_transformers":
            try:
                self._collection = chroma_client.get_or_create_collection(
                    name=name.replace(" ", "_"),
                    metadata={"hnsw:space": "cosine"},   # 論文指定 HNSW
                )
                self._mode = "chromadb"
            except Exception as e:
                print(f"  [VectorStore:{name}] ChromaDB init failed ({e})，使用記憶體模式")
        else:
            self._mode = "memory"

    def add_documents(self, docs: List[dict], text_field: str = "chunk"):
        """批量加入文件並向量化建立 HNSW 索引"""
        if not docs:
            return

        texts = [doc.get(text_field, "") for doc in docs]

        if self._mode == "chromadb" and self._collection is not None and _EMBED_MODE == "sentence_transformers":
            try:
                vecs = _embed(texts)
                ids        = [str(doc.get("id", f"{self.name}_{i}")) for i, doc in enumerate(docs)]
                metadatas  = [{"json": json.dumps(doc, ensure_ascii=False)} for doc in docs]
                # ChromaDB 1.5.x 要求純 Python float，不能是 numpy float32
                embeddings = [_to_float_list(v) for v in vecs]

                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                self._docs = docs
                print(f"  [ChromaDB:{self.name}] ✅ HNSW 索引建立 {len(docs)} 個語意分塊")
                return
            except Exception as e:
                print(f"  [ChromaDB:{self.name}] 索引失敗 ({e})，改用記憶體模式")
                self._mode = "memory"

        # Memory fallback（MiniLM 或 TF-IDF）
        self._docs = docs
        self._vecs = _embed(texts)
        print(f"  [VectorStore:{self.name}] 記憶體模式，已載入 {len(docs)} 個語意分塊")


    def query(
        self,
        query_text: str,
        top_k: int = 3,
        threshold: float = 0.75,
    ) -> List[Tuple[dict, float]]:
        """語意搜索，回傳 [(doc, similarity)] 列表"""

        if self._mode == "chromadb" and self._collection is not None:
            try:
                count = self._collection.count()
                if count == 0:
                    return []   # 空集合不查詢
                q_vec = _embed([query_text])[0]
                # 確保為純 Python float 列表，解決 ChromaDB 1.5.x 格式問題
                q_embedding = [_to_float_list(q_vec)]
                res = self._collection.query(
                    query_embeddings=q_embedding,
                    n_results=min(top_k, count),
                    include=["metadatas", "distances"],
                )
                results = []
                if res and res["ids"] and res["ids"][0]:
                    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
                        sim = max(0.0, 1.0 - float(dist))   # cosine distance → similarity
                        if sim >= threshold:
                            doc = json.loads(meta.get("json", "{}"))
                            results.append((doc, sim))
                return results[:top_k]
            except Exception as e:
                print(f"  [ChromaDB:{self.name}] query failed ({e})，使用記憶體搜索")

        # Memory fallback（MiniLM 或 TF-IDF）
        if not self._docs:
            return []
        q_vec = _embed([query_text])[0]
        scored = []
        for doc, vec in zip(self._docs, self._vecs):
            if _EMBED_MODE == "sentence_transformers":
                sim = _cosine_dense(q_vec, vec)
            else:
                sim = _cosine_sparse(q_vec, vec)
            scored.append((doc, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(d, s) for d, s in scored[:top_k] if s >= threshold]


    def get_all(self) -> List[dict]:
        return list(self._docs)

    def __len__(self):
        if self._mode == "chromadb" and self._collection:
            try:
                return self._collection.count()
            except Exception:
                pass
        return len(self._docs)


# ══════════════════════════════════════════════════════════════════════
# ChromaDB 客戶端初始化
# ══════════════════════════════════════════════════════════════════════

def _init_chroma(persist_dir: str = ".chroma_db"):
    """建立 ChromaDB PersistentClient（HNSW 索引持久化）"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)
        print(f"  [ChromaDB] ✅ PersistentClient 就緒（HNSW 索引路徑: {persist_dir}）")
        return client
    except Exception as e:
        print(f"  [ChromaDB] ⚠️  無法初始化 ({e})，使用記憶體模式")
        return None


# ══════════════════════════════════════════════════════════════════════
# 四大 RAG 資料庫
# ══════════════════════════════════════════════════════════════════════

class RAGDatabases:
    """
    論文第 4.1 節定義的四大 RAG 資料庫管理器。
    使用 ChromaDB + HNSW + sentence-transformers。
    """

    DB_FILES = {
        "environment": "environment_db.json",
        "rules":       "rules_db.json",
        "conditions":  "conditions_db.json",
        "targets":     "targets_db.json",
    }

    def __init__(self, db_dir: str = "db", persist_dir: str = ".chroma_db"):
        self.db_dir  = Path(db_dir)
        _load_embedding_model()               # 預載 embedding 模型

        # 初始化 ChromaDB
        self._chroma = _init_chroma(str(self.db_dir / persist_dir))

        self.stores: Dict[str, VectorStore] = {}
        self._load_all()

    def _load_all(self):
        print("\n[RAGDatabases] 載入四大 RAG 資料庫（ChromaDB + HNSW）...")
        for name, filename in self.DB_FILES.items():
            path = self.db_dir / filename
            store = VectorStore(name, chroma_client=self._chroma,
                                persist_dir=".chroma_db")
            try:
                docs = json.loads(path.read_text(encoding="utf-8"))
                store.add_documents(docs, text_field="chunk")
            except FileNotFoundError:
                print(f"  ⚠️  找不到: {path}")
            self.stores[name] = store

        total = self.total_chunks()
        print(f"[RAGDatabases] 就緒：共 {total} 個語意分塊\n")

    def total_chunks(self) -> int:
        return sum(len(s) for s in self.stores.values())

    def query_db(
        self, db_name: str, query_text: str,
        top_k: int = 3, threshold: float = 0.75,
    ) -> List[Tuple[dict, float]]:
        store = self.stores.get(db_name)
        return store.query(query_text, top_k=top_k, threshold=threshold) if store else []


# ══════════════════════════════════════════════════════════════════════
# RAG 引擎（完整管道，API 不變）
# ══════════════════════════════════════════════════════════════════════

class RAGEngine:
    """論文 4.1 節 RAG Implementation Pipeline（API 與舊版完全相容）"""

    TOP_K = 3
    SIMILARITY_THRESHOLD = 0.75

    def __init__(self, databases: Optional[RAGDatabases] = None, db_dir: str = "db"):
        self.dbs = databases or RAGDatabases(db_dir)

    def retrieve(
        self,
        task_instruction: str,
        current_state: Optional[dict] = None,
        target_area: str = "",
    ) -> Dict[str, List[Tuple[dict, float]]]:
        state_text    = " ".join(f"{k} {v}" for k, v in (current_state or {}).items())
        combined      = f"{task_instruction} {state_text} {target_area}".strip()
        print(f"  [RAGEngine] 混合查詢: «{combined[:80]}»")

        results = {}
        for db_name in self.dbs.stores:
            hits = self.dbs.query_db(
                db_name, combined,
                top_k=self.TOP_K,
                threshold=self.SIMILARITY_THRESHOLD,
            )
            results[db_name] = hits
            if hits:
                print(f"    📚 [{db_name}] 命中 {len(hits)} 條 (最高: {hits[0][1]:.3f})")
            else:
                print(f"    📚 [{db_name}] 無命中（閾值 {self.SIMILARITY_THRESHOLD}）")
        return results

    def build_constraints_block(
        self,
        retrieval_results: Dict[str, List[Tuple[dict, float]]],
    ) -> str:
        lines = ["[Constraints]"]
        label_map = {
            "environment": "🗺️ 環境", "rules": "📋 規則",
            "conditions": "⚙️ 條件", "targets": "🎯 目標",
        }
        for db_name, hits in retrieval_results.items():
            for doc, sim in hits:
                label = label_map.get(db_name, db_name)
                lines.append(f"  • [{label}] (sim:{sim:.2f}) {doc.get('chunk','')}")
        if len(lines) == 1:
            lines.append("  （無相關約束條件，按通用 SOP 執行）")
        return "\n".join(lines)

    def full_pipeline(
        self,
        task_instruction: str,
        current_state: Optional[dict] = None,
        target_area: str = "",
    ) -> Tuple[Dict, str]:
        results     = self.retrieve(task_instruction, current_state, target_area)
        constraints = self.build_constraints_block(results)
        return results, constraints

    def get_db_summary(self) -> str:
        lines = ["=== RAG 資料庫狀態 ==="]
        for name, store in self.dbs.stores.items():
            lines.append(f"  {name:15s}: {len(store):3d} 個語意分塊")
        lines.append(f"  {'合計':15s}: {self.dbs.total_chunks():3d} 個")
        return "\n".join(lines)


if __name__ == "__main__":
    engine = RAGEngine(db_dir="db")
    print(engine.get_db_summary())
    _, c = engine.full_pipeline(
        "前往森林區搜索可疑人員",
        {"Zone": "RT_FOR_01", "Event": "None"},
        "RT_FOR_01",
    )
    print(c)
