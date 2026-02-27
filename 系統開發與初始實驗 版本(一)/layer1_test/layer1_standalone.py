"""
layer1_standalone.py - Layer 1 Brain（論文符合版 v2）
=====================================================
免費生產級堆疊（不需任何付費 API）：
  ✅ ChromaDB + HNSW（論文指定 — 免費開源）
  ✅ paraphrase-multilingual-MiniLM-L6-v2（免費替代 text-embedding-3-small）
  ✅ Ollama + llava:13b（免費替代 GPT-4o）
  ✅ MockLLM（保留為 fallback）

完整的 Layer 1 決策管道（6 步驟）：
  步驟 0  情節記憶讀取      (Episodic Memory Retrieval — 論文 4.4 節)
  步驟 1  語意分塊與向量化  (Semantic Chunking & MiniLM Embedding)
  步驟 2  情境感知檢索      (Context-Aware Retrieval)
  步驟 3  Top-K 過濾        (K=3，閾值 0.75，fallback 0.5)
  步驟 4  動態 Prompt 注入  (Constraints + History)
  步驟 5  目標特徵動態生成  (Target Schema Generator — 論文 4.1 節)
  步驟 6  LLM 推論          (Ollama / MockLLM)
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

# ── RAG 引擎：優先使用 ChromaDB+MiniLM，fallback 到 TF-IDF ─────────────
try:
    from rag_engine import RAGEngine, RAGDatabases
    _RAG_BACKEND = "chromadb"
except Exception as _rag_err:
    print(f"  [Layer1] rag_engine 無法載入 ({_rag_err})，使用 TF-IDF fallback")
    from mock_rag import RAGEngine, RAGDatabases
    _RAG_BACKEND = "tfidf"

from mock_llm import get_llm_client
from episodic_memory import EpisodicMemory, seed_test_memories
from target_schema_generator import TargetSchemaGenerator



# ══════════════════════════════════════════════════════════════════════
# System Prompt 模板（論文 Prompt Engineering 設計）
# ══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """你是「基於仿生神經分工架構之生成式 AI 自主無人機系統」的核心決策大腦 (Layer 1 LLM Brain)。
你的職責是將自然語言任務指令轉化為符合 JSON Schema 的具體任務決策，交由 Layer 2 Behavior Planner 執行。

[你的能力]
- 語意理解：解析模糊任務指令，提取關鍵目標、區域與優先級
- 航點規劃：透過 Code-as-Planner 概念輸出具體航點序列（JSON 格式）
- 安全意識：所有決策必須符合 [Constraints] 中列出的規則與條件
- 歷史學習：參考 [History] 區塊中的過去失敗案例，避免重複錯誤
- 思維鏈：在 reasoning 欄位輸出推論過程

[輸出格式要求]
你的輸出必須是且只能是以下 JSON 格式：
{{
  "mission_id":           "L1_COMMAND_XXXXXX",
  "command":              "PATROL | RECON | SEARCH | RTH | HOVER",
  "target_area":          "RT_FOR_01 | RT_OPE_01 | RT_BUI_01 | RT_VEG_01",
  "waypoints":            [{{"x": float, "y": float, "z": float, "label": string}}, ...],
  "altitude":             float (NED 座標，負值為上升，例如 -10.0 = 飛行高度 10 公尺),
  "speed":                float (m/s，不得超過 5.0),
  "priority":             1-10,
  "timeout_sec":          float,
  "reasoning":            "推論說明（含 Chain-of-Thought），需說明為何做此決策",
  "constraints_applied":  [已套用的約束條目列表]
}}

[硬性限制]
- 速度上限：5.0 m/s（絕對不可超過）
- RT_BUI_01（建築禁航區）: z 必須 < -50m，或選擇繞飛
- 電量不足 20%：command 必須為 RTH，不接受其他指令
- [History] 中標記 ⚠️ 的事件代表過去失敗，請在 reasoning 中說明如何避免

{constraints_block}

{history_block}
"""


# ══════════════════════════════════════════════════════════════════════
# Layer 1 Brain 主控制器 v2
# ══════════════════════════════════════════════════════════════════════

class Layer1Brain:
    """
    Layer 1 Brain — 完整論文符合版 v2
    =====================================
    新增論文 4.4 節閉環學習機制：
      ・情節記憶 Retrieval（[History] 注入）
      ・動態 Target Schema 生成
      ・RAG 自適應閾值降級
    """

    RAG_THRESHOLD_PRIMARY  = 0.75   # 論文指定
    RAG_THRESHOLD_FALLBACK = 0.50   # 自適應降級（當主閾值無命中時）

    def __init__(
        self,
        db_dir:         str  = "db",
        use_openai:     bool = False,
        use_ollama:     bool = False,     # 免費替代 GPT-4o
        openai_api_key: str  = "",
        ollama_model:   str  = "",       # 留空=自動偵測(llava:13b)
        verbose:        bool = True,
        memory_path:    str  = "episodic_memory.json",
        bridge_memory:  str  = "",
        seed_memory:    bool = True,
    ):
        self.verbose = verbose

        print("=" * 62)
        print("  Layer 1 Brain v2 初始化中（論文符合度補全版）")
        print("=" * 62)

        # ── RAG 資料庫 ───────────────────────────────────────
        dbs = RAGDatabases(db_dir)
        self.rag = RAGEngine(databases=dbs)

        # ── LLM 客戶端 ──────────────────────────────────────
        self.llm = get_llm_client(
            use_openai=use_openai,
            use_ollama=use_ollama,
            api_key=openai_api_key,
            model=ollama_model,
            verbose=verbose,
        )

        # ── 情節記憶（論文 4.4 節）───────────────────────────
        bridge = bridge_memory if bridge_memory else None
        self.memory = EpisodicMemory(
            persist_path=memory_path,
            bridge_path=bridge,
        )
        if seed_memory and len(self.memory._store) == 0:
            seed_test_memories(self.memory)

        # ── 動態 Target Schema 生成器（論文 4.1 節）──────────
        self.target_gen = TargetSchemaGenerator(
            targets_db_path=f"{db_dir}/targets_db.json"
        )

        llm_mode = (
            "真實 OpenAI GPT-4o" if use_openai else
            f"Ollama ({getattr(self.llm, 'model', 'local')})" if use_ollama else
            "MockLLM（測試模式）"
        )
        print(f"\n  LLM 模式   : {llm_mode}")
        print(f"  RAG 後端   : {'ChromaDB+HNSW+MiniLM' if _RAG_BACKEND == 'chromadb' else 'TF-IDF'}")
        print(f"  {self.rag.get_db_summary()}")
        print(f"  {self.memory.summary()}")
        print("=" * 62)
        print("✅ Layer 1 Brain v2 初始化完成\n")


    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def decide(
        self,
        task_instruction: str,
        current_state:    Optional[dict] = None,
        target_area:      str  = "",
        battery_pct:      float = 100.0,
        generate_target:  bool  = True,    # 是否執行動態 Target Schema 生成
    ) -> dict:
        """
        Layer 1 完整決策管道 v2（論文 4.1+4.4 節）。

        步驟：
          0. 情節記憶讀取 (Episodic Memory Retrieval)  ← 新增
          1~3. RAG 四步驟管道（主閾值 0.75 → fallback 0.5）
          4. 動態 Prompt 注入（[Constraints] + [History]）
          5. 動態 Target Schema 生成                   ← 新增
          6. LLM 推論 → JSON 決策
        """
        t0 = time.time()

        if self.verbose:
            print(f"\n{'═'*62}")
            print(f"📡 [Layer1] 收到任務: 「{task_instruction[:60]}」")
            if current_state:
                print(f"   狀態: {current_state}")
            print(f"   電量: {battery_pct}%")

        # ── 電量安全優先（硬性規則）──────────────────────────
        if battery_pct < 20.0:
            print(f"🔴 [Layer1] 電量 {battery_pct}% < 20%，強制 RTH！")
            self.memory.log_event(
                zone=target_area or "Unknown",
                event_type="BatteryEmergency",
                decision="Forced RTH",
                outcome=f"Battery at {battery_pct:.0f}%",
                label="Negative",
            )
            return self._forced_rth(f"電量不足 {battery_pct:.0f}%")

        # ── 步驟 0：情節記憶讀取 ────────────────────────────
        if self.verbose:
            print(f"\n🧠 [步驟 0] 情節記憶讀取...")
        memory_query = f"{task_instruction} {target_area} " + (
            " ".join(f"{k} {v}" for k, v in (current_state or {}).items())
        )
        history_block = self.memory.build_history_block(memory_query, top_k=3)
        memories = self.memory.retrieve_experience(memory_query, top_k=3)
        if self.verbose:
            print(f"  → 找到 {len(memories)} 條相關歷史記憶")
            if memories:
                for m in memories:
                    label_icon = "⚠️" if m.get("label") == "Negative" else "✅"
                    print(f"    {label_icon} {m.get('event_type', '')} @ {m.get('zone', '')}: {m.get('outcome', '')[:50]}")

        # ── 步驟 1~3：RAG 管道（含自適應降級）─────────────────
        if self.verbose:
            print(f"\n📚 [步驟 1-3] RAG 管道（閾值 {self.RAG_THRESHOLD_PRIMARY}）...")
        retrieval_results, constraints_block = self._rag_with_fallback(
            task_instruction, current_state, target_area
        )

        # ── 步驟 4：動態雙區塊 Prompt 注入 ──────────────────
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            constraints_block=constraints_block,
            history_block=history_block,
        )
        if self.verbose:
            print(f"\n  [步驟 4] Prompt 注入完成")
            print(f"  [Constraints]\n{constraints_block}")
            print(f"\n  {history_block}")

        # ── 步驟 5：動態 Target Schema 生成 ─────────────────
        if generate_target:
            self._dynamic_target_generation(task_instruction)

        # ── 步驟 6：LLM 推論 ────────────────────────────────
        if self.verbose:
            print(f"\n🤖 [步驟 6] LLM 推論中...")
        raw = self.llm.chat_completion(system_prompt, task_instruction)

        # ── 解析與封裝 ──────────────────────────────────────
        decision = self._parse_json(raw)
        decision["_rag_hit_counts"]  = {db: len(hits) for db, hits in retrieval_results.items()}
        decision["_memory_hits"]     = len(memories)
        decision["_inference_ms"]    = round((time.time() - t0) * 1000, 1)

        if self.verbose:
            print(f"\n✅ 決策完成 ({decision['_inference_ms']}ms)")
            print(f"   {decision.get('command')} @ {decision.get('target_area')} | "
                  f"{len(decision.get('waypoints',[]))} 航點 | "
                  f"RAG 命中={decision['_rag_hit_counts']} | "
                  f"記憶命中={decision['_memory_hits']}")

        return decision

    def log_outcome(
        self,
        zone: str,
        event_type: str,
        decision: str,
        outcome: str,
        label: str = "Neutral",
        note: str = "",
    ):
        """
        記錄 Layer 1 決策結果到情節記憶（供下次決策參考）。
        Layer 2 也可以透過橋接路徑寫入。
        """
        self.memory.log_event(zone, event_type, decision, outcome, note, label)
        if self.verbose:
            icon = "⚠️" if label == "Negative" else "✅"
            print(f"  {icon} [記憶] 已記錄: {event_type} @ {zone} → {outcome[:50]}")

    def explain_constraints(self, task_instruction: str, current_state: Optional[dict] = None) -> str:
        """僅執行 RAG 管道，回傳 [Constraints] 區塊（不呼叫 LLM）"""
        _, constraints = self.rag.full_pipeline(task_instruction, current_state)
        return constraints

    def get_history(self, query: str = "") -> str:
        """回傳當前情節記憶 [History] 區塊"""
        return self.memory.build_history_block(query or "任務")

    def get_db_stats(self) -> str:
        return self.rag.get_db_summary()

    # ------------------------------------------------------------------
    # 內部方法
    # ------------------------------------------------------------------

    def _rag_with_fallback(
        self,
        task_instruction: str,
        current_state: Optional[dict],
        target_area: str,
    ) -> tuple:
        """
        RAG 管道 + 自適應閾值降級。
        若主閾值 (0.75) 無任何命中，自動降至 0.50 重試。
        """
        results, constraints = self.rag.full_pipeline(
            task_instruction, current_state, target_area
        )

        # 檢查是否有命中
        total_hits = sum(len(v) for v in results.values())
        if total_hits == 0:
            if self.verbose:
                print(f"  ⚠️  主閾值 {self.RAG_THRESHOLD_PRIMARY} 無命中，降級至 {self.RAG_THRESHOLD_FALLBACK}...")
            # 暫時降低閾值重試
            orig_threshold = self.rag.SIMILARITY_THRESHOLD
            self.rag.SIMILARITY_THRESHOLD = self.RAG_THRESHOLD_FALLBACK
            results, constraints = self.rag.full_pipeline(
                task_instruction, current_state, target_area
            )
            self.rag.SIMILARITY_THRESHOLD = orig_threshold  # 恢復
            total_hits = sum(len(v) for v in results.values())
            if self.verbose and total_hits > 0:
                print(f"  → 降級後命中 {total_hits} 條")

        return results, constraints

    def _dynamic_target_generation(self, instruction: str):
        """論文 4.1 節：動態 Target Schema 生成並 Upsert 到 targets_db"""
        # 只有當指令包含目標描述關鍵字時才生成
        target_kw = ["搜索", "尋找", "偵察", "find", "search", "target",
                      "人員", "車輛", "目標", "person", "vehicle"]
        if not any(kw in instruction for kw in target_kw):
            return

        if self.verbose:
            print(f"\n  [步驟 5] 動態 Target Schema 生成...")
        entry = self.target_gen.generate_from_instruction(instruction)
        success = self.target_gen.upsert_to_db(entry)
        if self.verbose and success:
            print(f"    → {entry['target_type']} | 顏色:{entry['colors']} | "
                  f"危險:{entry['danger_level']} | 信心度:{entry['confidence_threshold']:.0%}")

    def _parse_json(self, raw: str) -> dict:
        """安全解析 LLM 輸出 JSON"""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        return {
            "error": "JSON 解析失敗", "raw": raw,
            "command": "HOVER", "waypoints": [],
        }

    def _forced_rth(self, reason: str) -> dict:
        import uuid
        return {
            "mission_id":          f"L1_RTH_{uuid.uuid4().hex[:6].upper()}",
            "command":             "RTH",
            "target_area":         "HOME",
            "waypoints":           [{"x": 0, "y": 0, "z": -5, "label": "HomeBase"}],
            "altitude":            -5.0,
            "speed":               5.0,
            "priority":            10,
            "timeout_sec":         30.0,
            "reasoning":           f"強制 RTH：{reason}（論文 4.1 節電量安全規則）",
            "constraints_applied": ["RULE-BATT-06: 低電量返航"],
            "generated_by":        "Layer1_SafetyOverride",
            "_rag_hit_counts":     {},
            "_memory_hits":        0,
            "_inference_ms":       0,
        }


# ══════════════════════════════════════════════════════════════════════
# 快速測試
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    brain = Layer1Brain(db_dir="db", verbose=True)

    print("\n" + "═"*62)
    print("測試 1：Forest 偵蒐（含情節記憶 + 動態目標）")
    print("═"*62)
    d = brain.decide(
        task_instruction="前往森林區搜索綠衣可疑人員，注意避開建築禁航區",
        current_state={"Zone": "RT_FOR_01", "Event": "None"},
        target_area="RT_FOR_01",
        battery_pct=80.0,
    )
    print(f"\n📋 決策: {d.get('command')} @ {d.get('target_area')}")
    print(f"   記憶命中: {d.get('_memory_hits')} | RAG: {d.get('_rag_hit_counts')}")

    print("\n" + "═"*62)
    print("測試 2：記錄 Geofencing 負面事件後，下次決策行為改變")
    print("═"*62)
    brain.log_outcome(
        zone="RT_BUI_01",
        event_type="Geofencing_Blocked",
        decision="PATROL RT_BUI_01 z=-20",
        outcome="Layer 2 攔截：高度不足（z > -50m）",
        label="Negative",
        note="禁航區飛行高度必須低於 50m",
    )
    d2 = brain.decide(
        task_instruction="前往建築區附近偵蒐",
        current_state={"Zone": "RT_BUI_01"},
        target_area="RT_BUI_01",
        battery_pct=65.0,
    )
    print(f"\n📋 決策（有記憶參考）: {d2.get('command')} | reasoning 含記憶: {'建築' in d2.get('reasoning','') or '禁航' in d2.get('reasoning','')}")
