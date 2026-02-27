"""
test_layer2.py - Layer 2 Behavior Planner 自動化測試套件
==========================================================
測試覆蓋論文第 4.2 節所有核心機制：
  TC01 - FSM 狀態機合法轉換
  TC02 - FSM 非法轉換防護（競態條件保護）
  TC03 - Geofencing：合法航點通過
  TC04 - Geofencing：禁航區攔截（LLM 幻覺攔截）
  TC05 - Geofencing：未知座標攔截
  TC06 - 完整任務流程（Layer1 決策 → 路徑生成 → 執行）
  TC07 - 航點序列解析（Code-as-Planner）
  TC08 - 安全高度插入（急速爬升防護）
  TC09 - 斷線自主模式觸發
  TC10 - 任務逾時攔截
  TC11 - 情節記憶寫入與檢索
  TC12 - Target Found 訊號中斷任務
  TC13 - 空航點序列防護
  TC14 - 非法 JSON 輸入防護

執行方式：
  python test_layer2.py
  python test_layer2.py -v       （詳細輸出）
"""

import sys
import time
import threading
import unittest
from io import StringIO

from mock_memory import EpisodicMemory
from layer2_standalone import (
    FSM, FSMState,
    Geofence,
    PathPlanner,
    Waypoint,
    Layer1Decision,
    Layer2BehaviorPlanner,
)


# ══════════════════════════════════════════════════════════════════════
# 輔助工具
# ══════════════════════════════════════════════════════════════════════

def make_decision(
    waypoints: list,
    command: str = "PATROL",
    area: str = "Area1",
    timeout: float = 5.0,
) -> dict:
    """快速建立測試用 Layer1 決策 JSON"""
    return {
        "mission_id": f"TEST_{command}_{int(time.time())}",
        "command": command,
        "target_area": area,
        "waypoints": waypoints,
        "altitude": -10.0,
        "speed": 5.0,
        "priority": 5,
        "timeout_sec": timeout,
    }


def reset_fsm():
    """重設 FSM 單例（每個測試前呼叫）"""
    FSM().reset()


# ══════════════════════════════════════════════════════════════════════
# TC01-02  FSM 有限狀態機
# ══════════════════════════════════════════════════════════════════════

class TestFSM(unittest.TestCase):

    def setUp(self):
        reset_fsm()
        self.fsm = FSM()

    def test_tc01_valid_transitions(self):
        """TC01: FSM 合法轉換序列"""
        self.assertEqual(self.fsm.state, FSMState.IDLE)
        self.assertTrue(self.fsm.transition(FSMState.LOADING, "測試"))
        self.assertEqual(self.fsm.state, FSMState.LOADING)
        self.assertTrue(self.fsm.transition(FSMState.EXECUTING, "任務開始"))
        self.assertEqual(self.fsm.state, FSMState.EXECUTING)
        self.assertTrue(self.fsm.transition(FSMState.MISSION_DONE, "完成"))

    def test_tc02_invalid_transition_blocked(self):
        """TC02: 非法 FSM 轉換必須被攔截（防止競態條件）"""
        self.assertEqual(self.fsm.state, FSMState.IDLE)
        # IDLE → EXECUTING 是非法的（必須先經過 LOADING）
        result = self.fsm.transition(FSMState.EXECUTING, "跳過 LOADING")
        self.assertFalse(result)
        self.assertEqual(self.fsm.state, FSMState.IDLE)  # 狀態不應改變

    def test_tc02b_singleton_consistency(self):
        """TC02b: FSM 應為 Singleton（不同實例共享同一狀態）"""
        fsm_a = FSM()
        fsm_b = FSM()
        fsm_a.transition(FSMState.LOADING, "Singleton 測試")
        self.assertEqual(fsm_b.state, FSMState.LOADING)


# ══════════════════════════════════════════════════════════════════════
# TC03-05  地理圍欄
# ══════════════════════════════════════════════════════════════════════

class TestGeofence(unittest.TestCase):

    def setUp(self):
        self.geo = Geofence("environment.json")

    def test_tc03_valid_area_pass(self):
        """TC03: Area1 (Forest) 中的合法座標應通過"""
        is_safe, msg = self.geo.check(50.0, 50.0)   # Area1 範圍內
        self.assertTrue(is_safe, f"合法座標被攔截: {msg}")

    def test_tc04_high_danger_zone_blocked(self):
        """TC04: Area2 (Building/High Danger) 座標應被攔截 — 模擬 LLM 幻覺攔截"""
        is_safe, msg = self.geo.check(150.0, 150.0)  # Area2 範圍內
        self.assertFalse(is_safe, "高危區座標未被攔截！")
        self.assertIn("禁航區", msg)

    def test_tc05_unknown_coordinate_blocked(self):
        """TC05: 超出所有已知區域的座標應被攔截"""
        is_safe, msg = self.geo.check(999.0, 999.0)
        self.assertFalse(is_safe, "未知座標未被攔截！")

    def test_tc03b_batch_validation(self):
        """TC03b: 批量驗證，混合合法/非法航點"""
        wps = [
            Waypoint(50, 50),   # Area1 合法
            Waypoint(150, 150), # Area2 禁航
        ]
        results = self.geo.validate_waypoints(wps)
        self.assertEqual(len(results), 2)
        _, safe1, _ = results[0]
        _, safe2, _ = results[1]
        self.assertTrue(safe1)
        self.assertFalse(safe2)


# ══════════════════════════════════════════════════════════════════════
# TC07-08  路徑規劃器
# ══════════════════════════════════════════════════════════════════════

class TestPathPlanner(unittest.TestCase):

    def test_tc07_waypoint_parsing(self):
        """TC07: Code-as-Planner — Layer1 JSON 航點正確解析"""
        decision = Layer1Decision.from_json(make_decision(
            waypoints=[
                {"x": 10, "y": 20, "label": "Alpha"},
                {"x": 30, "y": 40},
                {"x": 50, "y": 60, "z": -15},
            ]
        ))
        wps = PathPlanner.from_layer1_decision(decision)
        self.assertEqual(len(wps), 3)
        self.assertEqual(wps[0].x, 10)
        self.assertEqual(wps[0].label, "Alpha")
        self.assertEqual(wps[2].z, -15)  # 明確指定高度

    def test_tc07b_invalid_waypoint_skipped(self):
        """TC07b: 格式錯誤的航點應被跳過，不影響其他航點"""
        decision = Layer1Decision.from_json(make_decision(
            waypoints=[
                {"x": 10, "y": 20},
                {"BAD": "DATA"},         # 無效
                {"x": 30, "y": 40},
            ]
        ))
        wps = PathPlanner.from_layer1_decision(decision)
        self.assertEqual(len(wps), 2)    # 有效航點數應為 2

    def test_tc08_safe_altitude_insertion(self):
        """TC08: 急速爬升時應自動插入安全高度中間點"""
        wps = [
            Waypoint(0, 0, -5.0),
            Waypoint(50, 50, -30.0),  # 高度差 25m > 閾值 5m
        ]
        result = PathPlanner.insert_safe_altitude(wps, safe_z=-20.0, climb_threshold=5.0)
        self.assertGreater(len(result), 2, "應插入安全高度中間點")
        mid = result[1]
        self.assertEqual(mid.z, -20.0)
        self.assertEqual(mid.label, "AUTO_CLIMB_POINT")

    def test_tc08b_no_insertion_when_safe(self):
        """TC08b: 高度差小於閾值時，不應插入額外航點"""
        wps = [Waypoint(0, 0, -10.0), Waypoint(10, 10, -12.0)]
        result = PathPlanner.insert_safe_altitude(wps)
        self.assertEqual(len(result), 2)

    def test_tc07c_distance_calculation(self):
        """TC07c: 總距離計算正確性"""
        wps = [Waypoint(0, 0), Waypoint(3, 4)]  # 距離 = 5
        dist = PathPlanner.compute_total_distance(wps)
        self.assertAlmostEqual(dist, 5.0, places=5)


# ══════════════════════════════════════════════════════════════════════
# TC06, TC09-14  Layer2 整合測試
# ══════════════════════════════════════════════════════════════════════

class TestLayer2Integration(unittest.TestCase):

    def setUp(self):
        reset_fsm()
        self.memory = EpisodicMemory()
        self.layer2 = Layer2BehaviorPlanner(
            env_path="environment.json",
            memory=self.memory,
        )

    def tearDown(self):
        self.layer2.stop()

    def test_tc06_full_mission_flow(self):
        """TC06: 完整任務流程 — Layer1 決策 → Geofence → 路徑生成 → EXECUTING"""
        decision_json = make_decision(
            waypoints=[
                {"x": 10, "y": 20, "label": "WP_A"},
                {"x": 50, "y": 50, "label": "WP_B"},
                {"x": 80, "y": 30, "label": "WP_C"},
            ],
            area="Area1",
        )
        result = self.layer2.process_layer1_decision(decision_json)
        self.assertTrue(result, "合法任務應成功啟動")
        self.assertEqual(self.layer2.fsm.state, FSMState.EXECUTING)

    def test_tc09_autonomous_mode_activation(self):
        """TC09: 斷線時切換自主模式"""
        # 先寫入一些歷史記憶
        self.memory.write_event("PATROL", "Area1 巡邏成功", severity="INFO")

        self.layer2.activate_autonomous_mode("Layer1 API 逾時")
        self.assertEqual(self.layer2.fsm.state, FSMState.AUTONOMOUS)

    def test_tc10_geofence_blocks_mission(self):
        """TC10: 含禁航區航點的任務應被 Geofence 攔截"""
        decision_json = make_decision(
            waypoints=[
                {"x": 10, "y": 20},     # Area1 合法
                {"x": 150, "y": 150},   # Area2 禁航！
            ],
            area="Area2",
        )
        result = self.layer2.process_layer1_decision(decision_json)
        self.assertFalse(result, "含禁航座標的任務應被攔截")
        self.assertEqual(self.layer2.fsm.state, FSMState.ABORT)

    def test_tc11_episodic_memory_write(self):
        """TC11: 任務事件應被正確寫入情節記憶"""
        initial_count = len(self.memory.get_all_events())
        decision_json = make_decision(
            waypoints=[{"x": 10, "y": 20}],
        )
        self.layer2.process_layer1_decision(decision_json)
        time.sleep(0.1)
        new_count = len(self.memory.get_all_events())
        self.assertGreater(new_count, initial_count, "應有新事件被寫入記憶庫")

    def test_tc11b_memory_retrieval(self):
        """TC11b: 情節記憶檢索 — 應返回相關負面經驗"""
        self.memory.write_event("GEOFENCE_VIOLATION", "嘗試飛入 Area2 禁航區", severity="WARNING")
        self.memory.write_event("TARGET_FOUND", "在 Area1 發現目標", severity="INFO")

        lessons = self.memory.retrieve_lessons_learned("Area2 禁航區")
        self.assertTrue(any("GEOFENCE" in l or "Area2" in l for l in lessons))

    def test_tc12_target_found_hover(self):
        """TC12: 收到 Target Found 訊號應觸發懸停"""
        # 先啟動任務讓 FSM 進入 EXECUTING
        decision_json = make_decision(
            waypoints=[{"x": 50, "y": 50}],
        )
        self.layer2.process_layer1_decision(decision_json)
        self.assertEqual(self.layer2.fsm.state, FSMState.EXECUTING)

        # 送出 Target Found
        self.layer2.signal_target_found(confidence=0.95)
        self.assertEqual(self.layer2.fsm.state, FSMState.HOVERING)

    def test_tc13_empty_waypoints_rejected(self):
        """TC13: 空航點序列應被安全拒絕"""
        decision_json = make_decision(waypoints=[])
        result = self.layer2.process_layer1_decision(decision_json)
        self.assertFalse(result, "空航點任務應被拒絕")

    def test_tc14_malformed_json_rejected(self):
        """TC14: 缺少必要欄位的 JSON 應被安全處理（不應崩潰）"""
        bad_json = {"totally": "wrong", "no": "waypoints"}
        try:
            result = self.layer2.process_layer1_decision(bad_json)
            # 要麼回傳 False，要麼不崩潰
            self.assertFalse(result)
        except SystemExit:
            self.fail("惡意輸入造成程式崩潰！")


# ══════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Layer 2 Behavior Planner — 自動化測試套件")
    print("  論文：基於仿生神經分工架構之生成式 AI 自主無人機研究")
    print("=" * 60)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for cls in [TestFSM, TestGeofence, TestPathPlanner, TestLayer2Integration]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    result = runner.run(suite)

    print()
    if result.wasSuccessful():
        print("✅ 所有測試通過！Layer 2 核心機制驗證完成。")
    else:
        print(f"❌ {len(result.failures)} 項失敗, {len(result.errors)} 項錯誤")
        sys.exit(1)


if __name__ == "__main__":
    main()
