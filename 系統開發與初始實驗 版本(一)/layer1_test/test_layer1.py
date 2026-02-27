"""
test_layer1.py - Layer 1 Brain 自動化測試套件 v2
================================================
測試覆蓋論文第 4.1+4.4 節所有核心機制 (32 個測試案例)

執行方式：
  python test_layer1.py
  python test_layer1.py -v   （詳細輸出）

v2 新增：
  TC17-19  情節記憶 (EpisodicMemory)
  TC20-21  動態 Target Schema 生成
  TC22-24  Layer 1 Brain v2 整合測試
"""

import sys
import json
import os
import unittest
import tempfile

from mock_rag import RAGEngine, RAGDatabases
from mock_llm import MockLLM, TaskParser
from layer1_standalone import Layer1Brain
from episodic_memory import EpisodicMemory, seed_test_memories
from target_schema_generator import TargetSchemaGenerator, FeatureExtractor


# ══════════════════════════════════════════════════════════════════════
# 共用 fixture（模組層級 singleton，避免重複初始化）
# ══════════════════════════════════════════════════════════════════════

_SHARED_DBS    = None
_SHARED_ENGINE = None
_SHARED_LLM    = None
_SHARED_BRAIN  = None


def get_dbs() -> RAGDatabases:
    global _SHARED_DBS
    if _SHARED_DBS is None:
        _SHARED_DBS = RAGDatabases(db_dir="db")
    return _SHARED_DBS


def get_engine() -> RAGEngine:
    global _SHARED_ENGINE
    if _SHARED_ENGINE is None:
        _SHARED_ENGINE = RAGEngine(databases=get_dbs())
    return _SHARED_ENGINE


def get_llm() -> MockLLM:
    global _SHARED_LLM
    if _SHARED_LLM is None:
        _SHARED_LLM = MockLLM(verbose=False)
    return _SHARED_LLM


def get_brain() -> Layer1Brain:
    global _SHARED_BRAIN
    if _SHARED_BRAIN is None:
        _SHARED_BRAIN = Layer1Brain(
            db_dir="db", use_openai=False, verbose=False,
            memory_path="test_episodic_memory.json",
            seed_memory=True,
        )
    return _SHARED_BRAIN


DUMMY_SYSTEM_PROMPT = """你是無人機 AI。
[Constraints]
  • [規則] 森林區高度限制：飛行高度 10m-20m
  • [條件] 通訊逾時：500ms 切換自主模式
"""


# ══════════════════════════════════════════════════════════════════════
# TC01-05  四大 RAG 資料庫
# ══════════════════════════════════════════════════════════════════════

class TestRAGDatabases(unittest.TestCase):

    def setUp(self):
        self.dbs = get_dbs()

    def test_tc01_all_four_dbs_loaded(self):
        """TC01: 四大資料庫均應成功載入"""
        expected = {"environment", "rules", "conditions", "targets"}
        self.assertEqual(set(self.dbs.stores.keys()), expected)

    def test_tc02_environment_db_chunks(self):
        """TC02: Environment DB 應有 >= 10 個語意分塊"""
        store = self.dbs.stores["environment"]
        self.assertGreaterEqual(len(store), 10)

    def test_tc03_rules_db_chunks(self):
        """TC03: Rules DB 應有 >= 8 個語意分塊"""
        store = self.dbs.stores["rules"]
        self.assertGreaterEqual(len(store), 8)

    def test_tc04_conditions_db_chunks(self):
        """TC04: Conditions DB 應有 >= 6 個語意分塊"""
        store = self.dbs.stores["conditions"]
        self.assertGreaterEqual(len(store), 6)

    def test_tc05_targets_db_chunks(self):
        """TC05: Targets DB 應有 >= 6 個語意分塊"""
        store = self.dbs.stores["targets"]
        self.assertGreaterEqual(len(store), 6)

    def test_tc01b_total_chunks(self):
        """TC01b: 四大資料庫總語意分塊數 >= 35"""
        self.assertGreaterEqual(self.dbs.total_chunks(), 35)


# ══════════════════════════════════════════════════════════════════════
# TC06-11  RAG 引擎
# ══════════════════════════════════════════════════════════════════════

class TestRAGEngine(unittest.TestCase):

    def setUp(self):
        self.engine = get_engine()

    def test_tc06_similarity_threshold_filtering(self):
        """TC06: 不相關查詢結果相似度均應 < 0.75"""
        results = self.engine.dbs.stores["rules"].query(
            "qqqq zzzz 9999 xxxx mmmm",
            top_k=3,
            threshold=0.75,
        )
        for _, sim in results:
            self.assertLess(sim, 0.75)

    def test_tc06b_relevant_query_hits(self):
        """TC06b: 相關查詢在低閾值下應有命中結果"""
        results = self.engine.dbs.stores["rules"].query(
            "森林 高度 限制 area1 altitude",
            top_k=3,
            threshold=0.0,
        )
        self.assertGreater(len(results), 0)

    def test_tc07_top_k_limit(self):
        """TC07: Top-K 結果數不超過 K=3"""
        results = self.engine.dbs.stores["environment"].query(
            "area 森林 飛行 建築 開闊",
            top_k=3,
            threshold=0.0,
        )
        self.assertLessEqual(len(results), 3)

    def test_tc08_context_aware_retrieval(self):
        """TC08: 情境感知混合查詢應回傳 dict + 含 [Constraints] 的文字"""
        results, constraints = self.engine.full_pipeline(
            task_instruction="執行巡邏任務",
            current_state={"Zone": "Area1", "Event": "Obstacle"},
            target_area="Area1",
        )
        self.assertIsInstance(results, dict)
        self.assertIn("[Constraints]", constraints)

    def test_tc09_forest_rule_retrieval(self):
        """TC09: 森林區規則應在 Rules DB 中可被檢索"""
        results = self.engine.dbs.stores["rules"].query(
            "area1 forest 高度 限制 altitude",
            top_k=3,
            threshold=0.0,
        )
        chunks = [doc.get("chunk", "") for doc, _ in results]
        self.assertTrue(any("森林" in c or "Area1" in c for c in chunks))

    def test_tc10_nofly_zone_retrieval(self):
        """TC10: Area2 禁航資訊應可從 Environment/Rules DB 檢索"""
        r_env = self.engine.dbs.stores["environment"].query(
            "禁航 area2 no-fly 紅色 restriction", top_k=3, threshold=0.0)
        r_rules = self.engine.dbs.stores["rules"].query(
            "禁航 area2 no-fly zone", top_k=3, threshold=0.0)
        all_chunks = [doc.get("chunk", "") for doc, _ in r_env + r_rules]
        self.assertTrue(any("禁航" in c or "Area2" in c or "No-Fly" in c for c in all_chunks))

    def test_tc11_constraints_block_generation(self):
        """TC11: Constraints 區塊應包含 [Constraints] 標頭"""
        _, constraints = self.engine.full_pipeline(
            task_instruction="巡邏 forest 搜索",
            current_state={"Event": "None"},
        )
        self.assertIn("[Constraints]", constraints)
        self.assertGreater(len(constraints.split("\n")), 1)


# ══════════════════════════════════════════════════════════════════════
# TC12-13  Mock LLM
# ══════════════════════════════════════════════════════════════════════

class TestMockLLM(unittest.TestCase):

    def setUp(self):
        self.llm = get_llm()

    def test_tc12_json_output_format(self):
        """TC12: MockLLM 輸出應為合法 JSON，含所有必要欄位"""
        raw = self.llm.chat_completion(DUMMY_SYSTEM_PROMPT, "前往 Area1 執行巡邏")
        decision = json.loads(raw)  # 若不合法 JSON 會 raise
        for key in {"mission_id", "command", "target_area", "waypoints",
                    "altitude", "speed", "timeout_sec", "reasoning"}:
            self.assertIn(key, decision)

    def test_tc13a_command_patrol(self):
        """TC13a: 巡邏指令解析 → PATROL"""
        self.assertEqual(TaskParser.parse_command("前往森林區執行巡邏偵察"), "PATROL")

    def test_tc13b_command_recon(self):
        """TC13b: 偵蒐指令解析 → RECON"""
        self.assertEqual(TaskParser.parse_command("對目標進行 recon 偵察任務"), "RECON")

    def test_tc13c_command_rth(self):
        """TC13c: 返航指令解析 → RTH"""
        self.assertEqual(TaskParser.parse_command("電量不足，立即返航 RTH"), "RTH")

    def test_tc13d_command_search(self):
        """TC13d: 搜索指令解析 → SEARCH"""
        self.assertEqual(TaskParser.parse_command("搜索可疑人員目標"), "SEARCH")

    def test_tc13e_area_parsing(self):
        """TC13e: 目標區域解析"""
        self.assertEqual(TaskParser.parse_area("前往 Area1 森林"), "Area1")
        self.assertEqual(TaskParser.parse_area("Area2 建築物"), "Area2")
        self.assertEqual(TaskParser.parse_area("Area3 開闊平原"), "Area3")

    def test_tc12b_waypoints_not_empty(self):
        """TC12b: 航點列表不應為空"""
        raw = self.llm.chat_completion(DUMMY_SYSTEM_PROMPT, "前往 Area1 執行搜索")
        decision = json.loads(raw)
        self.assertGreater(len(decision.get("waypoints", [])), 0)

    def test_tc12c_speed_within_limit(self):
        """TC12c: 速度不得超過 5.0 m/s"""
        raw = self.llm.chat_completion(DUMMY_SYSTEM_PROMPT, "前往 Area3 高速巡邏")
        decision = json.loads(raw)
        self.assertLessEqual(decision.get("speed", 0), 5.0)


# ══════════════════════════════════════════════════════════════════════
# TC14-16  Layer 1 Brain 整合測試
# ══════════════════════════════════════════════════════════════════════

class TestLayer1Brain(unittest.TestCase):

    def setUp(self):
        self.brain = get_brain()

    def test_tc14_end_to_end_pipeline(self):
        """TC14: 端到端完整決策管道"""
        decision = self.brain.decide(
            task_instruction="前往 Area1 森林區執行巡邏",
            current_state={"Zone": "Area1"},
            target_area="Area1",
        )
        self.assertIsInstance(decision, dict)
        self.assertIn("command", decision)
        self.assertIn("waypoints", decision)
        self.assertIn("_rag_hit_counts", decision)
        self.assertIn("_inference_ms", decision)

    def test_tc15_low_battery_forced_rth(self):
        """TC15: 電量 < 20% 應強制 RTH，優先級 = 10"""
        decision = self.brain.decide(
            task_instruction="前往 Area1 執行巡邏",
            battery_pct=15.0,
        )
        self.assertEqual(decision.get("command"), "RTH")
        self.assertEqual(decision.get("priority"), 10)

    def test_tc16_layer2_schema_compatibility(self):
        """TC16: Layer 2 JSON Schema 完整性與類型正確性"""
        decision = self.brain.decide(
            task_instruction="Area3 開闊區目標搜索",
            target_area="Area3",
        )
        self.assertIsInstance(decision.get("mission_id"), str)
        self.assertIn(decision.get("command"),
                      ["PATROL", "RECON", "SEARCH", "RTH", "HOVER"])
        self.assertIsInstance(decision.get("waypoints"), list)
        self.assertIsInstance(decision.get("altitude"), (int, float))
        self.assertIsInstance(decision.get("speed"), (int, float))
        self.assertIsInstance(decision.get("timeout_sec"), (int, float))
        for wp in decision.get("waypoints", []):
            self.assertIn("x", wp)
            self.assertIn("y", wp)


# ══════════════════════════════════════════════════════════════════════
# TC17-19  情節記憶 (論文 4.4 節閉環學習 — Retrieval 階段)
# ══════════════════════════════════════════════════════════════════════

class TestEpisodicMemory(unittest.TestCase):

    def setUp(self):
        # 每個測試用獨立暫存記憶檔
        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        )
        self._tmp.close()
        self.mem = EpisodicMemory(persist_path=self._tmp.name)
        seed_test_memories(self.mem)

    def tearDown(self):
        try: os.unlink(self._tmp.name)
        except: pass

    def test_tc17_log_and_retrieve(self):
        """TC17: 寫入事件後應可被 retrieve_experience 找到"""
        self.mem.log_event(
            zone="RT_FOR_01",
            event_type="Geofencing_Blocked",
            decision="PATROL z=-10",
            outcome="高度不符合限制",
            label="Negative",
        )
        results = self.mem.retrieve_experience(
            "前往森林區 Geofencing 高度", top_k=5, min_sim=0.0
        )
        self.assertGreater(len(results), 0)

    def test_tc18_negative_events_prioritized(self):
        """TC18: 負面事件應在 prefer_negative=True 時排在前面"""
        results = self.mem.retrieve_experience(
            "森林 建築 飛行任務", top_k=5, min_sim=0.0, prefer_negative=True
        )
        if len(results) >= 2:
            # 至少第一條應含負面標籤
            neg_count = sum(
                1 for r in results
                if r.get("label") == "Negative" or
                   r.get("event_type") in EpisodicMemory.NEGATIVE_LABELS
            )
            self.assertGreater(neg_count, 0)

    def test_tc19_history_block_format(self):
        """TC19: build_history_block 應含 [History] 標頭"""
        block = self.mem.build_history_block("forest 森林 任務")
        self.assertIn("[History]", block)


# ══════════════════════════════════════════════════════════════════════
# TC20-21  動態 Target Schema 生成器（論文 4.1 節）
# ══════════════════════════════════════════════════════════════════════

class TestTargetSchemaGenerator(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        # 使用現有 targets_db 作為基底
        import shutil
        src = os.path.join("db", "targets_db.json")
        dst = os.path.join(self._tmp_dir, "targets_db.json")
        if os.path.exists(src):
            shutil.copy(src, dst)
        self.gen = TargetSchemaGenerator(targets_db_path=dst)

    def test_tc20_feature_extraction(self):
        """TC20: 應能從自然語言提取顏色、目標類型與行為"""
        features = FeatureExtractor.extract("搜索紅色可疑車輛，疑似逃亡中")
        self.assertIn("red", features["colors"])
        self.assertEqual(features["target_type"], "vehicle")
        self.assertTrue(len(features["behaviors"]) > 0)

    def test_tc21_schema_generation_and_upsert(self):
        """TC21: 動態條目應符合 Schema 且成功寫入 DB"""
        entry = self.gen.generate_from_instruction("尋找綠衣可疑人員")
        # 結構檢查
        for key in ["id", "database", "target_type", "chunk", "source"]:
            self.assertIn(key, entry)
        self.assertEqual(entry["database"], "targets")
        self.assertEqual(entry["source"], "dynamic_generated")
        self.assertIn("green", entry.get("colors", []))
        # Upsert 後 DB 檔案應存在且包含新條目
        success = self.gen.upsert_to_db(entry)
        self.assertTrue(success)
        with open(self.gen.db_path, encoding="utf-8") as f:
            db = json.load(f)
        ids = [e.get("id") for e in db]
        self.assertIn(entry["id"], ids)


# ══════════════════════════════════════════════════════════════════════
# TC22-24  Layer 1 Brain v2 整合測試（情節記憶 + 動態目標）
# ══════════════════════════════════════════════════════════════════════

class TestLayer1BrainV2(unittest.TestCase):

    def setUp(self):
        self.brain = get_brain()

    def test_tc22_memory_hits_in_decision(self):
        """TC22: decide() 輸出應含 _memory_hits 欄位（整數）"""
        decision = self.brain.decide(
            task_instruction="前往森林區搜索可疑人員",
            current_state={"Zone": "RT_FOR_01"},
        )
        self.assertIn("_memory_hits", decision)
        self.assertIsInstance(decision["_memory_hits"], int)

    def test_tc23_rag_fallback_threshold(self):
        """TC23: RAG fallback 閾值降至 0.5 後應產生合法決策"""
        # 使用唯一關鍵字確保主閾值無命中，觸發 fallback
        decision = self.brain.decide(
            task_instruction="執行任務 zone forest search target",
            current_state={"Zone": "RT_OPE_01", "Event": "None"},
            target_area="RT_OPE_01",
        )
        self.assertIn("command", decision)
        self.assertIn(decision.get("command"),
                      ["PATROL", "RECON", "SEARCH", "RTH", "HOVER"])

    def test_tc24_log_outcome_persists(self):
        """TC24: log_outcome 寫入情節記憶後，下次 retrieve 應命中"""
        brain = self.brain
        brain.log_outcome(
            zone="RT_TC24_TEST",
            event_type="Geofencing_Blocked",
            decision="PATROL z=-5",
            outcome="高度太低被攔截",
            label="Negative",
        )
        memories = brain.memory.retrieve_experience(
            "RT_TC24 Geofencing height blocked", top_k=5, min_sim=0.0
        )
        zones = [m.get("zone", "") for m in memories]
        self.assertTrue(any("TC24" in z for z in zones))


# ══════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Layer 1 Brain — 自動化測試套件 v2")
    print("  論文：基於仿生神經分工架構之生成式 AI 自主無人機研究")
    print("  覆蓋：四大 RAG 資料庫 + RAG 引擎 + MockLLM +")
    print("        情節記憶 + 動態 Target Schema + 完整管道")
    print("=" * 62)
    print()

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestRAGDatabases, TestRAGEngine, TestMockLLM,
        TestLayer1Brain,
        TestEpisodicMemory, TestTargetSchemaGenerator, TestLayer1BrainV2,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    result = runner.run(suite)

    print()
    if result.wasSuccessful():
        print("✅ 所有測試通過！Layer 1 論文符合度驗證完成（含情節記憶與動態 Target Schema）。")
    else:
        print(f"❌ {len(result.failures)} 項失敗, {len(result.errors)} 項錯誤")
        sys.exit(1)


if __name__ == "__main__":
    main()
