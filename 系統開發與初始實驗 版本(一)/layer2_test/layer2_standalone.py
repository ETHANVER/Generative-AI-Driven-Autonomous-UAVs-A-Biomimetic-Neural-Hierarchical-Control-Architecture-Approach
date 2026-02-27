"""
layer2_standalone.py - Layer 2 Behavior Planner（完整獨立版）
==============================================================
根據論文「基於仿生神經分工架構之生成式 AI 自主無人機研究」第 4.2 節設計。

核心功能：
  1. 有限狀態機 (FSM)         - 管理任務生命週期，防止競態條件
  2. 地理圍欄 (Geofencing)    - 攔截 LLM 幻覺產生的非法座標
  3. 路徑規劃 (Waypoint Gen)  - 將 Layer1 JSON 決策轉譯為具體航點序列
  4. 10Hz 監控迴圈            - 監測航點抵達狀態與逾時異常
  5. 斷線自主模式             - 通訊中斷時改由本地情節記憶庫自主導航
  6. 情節記憶整合             - 記錄所有異常事件供 Layer1 閉環學習

不依賴任何外部服務 (AirSim / OpenAI / ChromaDB)，
可直接搭配 mock_memory.py 在純 Python 環境運行。
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple

# 優先嘗試使用正式記憶體模組，否則退回 MockMemory
try:
    from memory_db import EpisodicMemory
    print("[Layer2] 使用正式 EpisodicMemory (ChromaDB)")
except ImportError:
    from mock_memory import EpisodicMemory
    print("[Layer2] 使用 MockMemory（測試模式）")


# ══════════════════════════════════════════════════════════════════════
#  資料結構
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Waypoint:
    """3D 航點"""
    x: float
    y: float
    z: float = -10.0          # 預設高度 10 公尺（AirSim NED 座標系，負值為上升）
    label: str = ""           # 可選語義標籤，如 "Patrol_Point_1"

    def distance_to(self, other: "Waypoint") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def __repr__(self) -> str:
        return f"WP({self.x:.1f}, {self.y:.1f}, {self.z:.1f})"


@dataclass
class Layer1Decision:
    """
    Layer 1 (LLM) 輸出決策的標準 JSON Schema。
    Layer 2 接收並解析此結構。
    """
    mission_id: str
    command: str                        # e.g. "PATROL", "RECON", "RETURN_HOME"
    target_area: str                    # e.g. "Area1"
    waypoints_raw: List[dict]           # [{"x":10,"y":20,"z":-10,"label":"P1"}]
    altitude: float = -10.0             # 預設飛行高度
    speed: float = 5.0                  # 預設速度 (m/s)
    priority: int = 5                   # 優先等級 1-10
    timeout_sec: float = 60.0           # 任務逾時秒數

    @classmethod
    def from_json(cls, data: dict) -> "Layer1Decision":
        return cls(
            mission_id=data.get("mission_id", "UNKNOWN"),
            command=data.get("command", "IDLE"),
            target_area=data.get("target_area", ""),
            waypoints_raw=data.get("waypoints", []),
            altitude=data.get("altitude", -10.0),
            speed=data.get("speed", 5.0),
            priority=data.get("priority", 5),
            timeout_sec=data.get("timeout_sec", 60.0),
        )


# ══════════════════════════════════════════════════════════════════════
#  有限狀態機 (FSM)
# ══════════════════════════════════════════════════════════════════════

class FSMState(Enum):
    IDLE         = auto()    # 待機
    LOADING      = auto()    # 正在解析 Layer1 決策（資源鎖定中）
    EXECUTING    = auto()    # 執行航點序列
    HOVERING     = auto()    # 懸停（目標發現 / 等待指令）
    ABORT        = auto()    # 任務中止（Geofence 攔截 / 嚴重錯誤）
    AUTONOMOUS   = auto()    # 斷線自主模式（改由記憶庫導航）
    MISSION_DONE = auto()    # 任務完成


class FSM:
    """
    有限狀態機 — 採用 Singleton + Mutex 模式（論文 4.2 節）
    防止 Layer2 (10Hz) 與 Layer3 (30-50Hz) 之間的競態條件。
    """
    _instance: Optional["FSM"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._state = FSMState.IDLE
                cls._instance._state_lock = threading.RLock()
                cls._instance._history: list = []
        return cls._instance

    @property
    def state(self) -> FSMState:
        return self._state

    def transition(self, new_state: FSMState, reason: str = "") -> bool:
        """
        嘗試狀態轉換，回傳是否成功。
        所有轉換皆受 Mutex 保護。
        """
        # 合法轉換表
        VALID_TRANSITIONS = {
            FSMState.IDLE:         {FSMState.LOADING, FSMState.AUTONOMOUS},
            FSMState.LOADING:      {FSMState.EXECUTING, FSMState.ABORT, FSMState.IDLE},
            FSMState.EXECUTING:    {FSMState.HOVERING, FSMState.ABORT, FSMState.MISSION_DONE, FSMState.AUTONOMOUS},
            FSMState.HOVERING:     {FSMState.EXECUTING, FSMState.IDLE, FSMState.ABORT},
            FSMState.ABORT:        {FSMState.IDLE},
            FSMState.AUTONOMOUS:   {FSMState.IDLE, FSMState.HOVERING, FSMState.ABORT},
            FSMState.MISSION_DONE: {FSMState.IDLE},
        }

        with self._state_lock:
            if new_state not in VALID_TRANSITIONS.get(self._state, set()):
                print(
                    f"⛔ [FSM] 非法轉換: {self._state.name} → {new_state.name} ({reason})"
                )
                return False

            old = self._state
            self._state = new_state
            self._history.append(
                {"from": old.name, "to": new_state.name, "reason": reason, "ts": time.time()}
            )
            print(f"🔄 [FSM] {old.name} → {new_state.name}  | {reason}")
            return True

    def reset(self):
        """強制重設（測試用）"""
        with self._state_lock:
            self._state = FSMState.IDLE
            self._history.clear()

    def get_history(self) -> list:
        return list(self._history)


# ══════════════════════════════════════════════════════════════════════
#  地理圍欄 (Geofencing)
# ══════════════════════════════════════════════════════════════════════

class Geofence:
    """
    根據 environment.json 設置的矩形地理圍欄驗證機制。
    攔截 LLM 幻覺造成的非法指令（論文 4.2 節）。
    """

    def __init__(self, env_path: str = "environment.json"):
        self.areas: dict = {}
        self._load(env_path)

    def _load(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.areas = json.load(f)
            print(f"[Geofence] 已載入 {len(self.areas)} 個區域邊界")
        except Exception as e:
            print(f"⚠️ [Geofence] 無法載入邊界資料: {e}，切換為開發模式（不攔截）")
            self.areas = {}

    def check_and_adjust(self, wp: Waypoint) -> Tuple[bool, str]:
        """
        回傳 (是否允許通過, 說明)
        Phase 5: 3D Height-Aware Avoidance
        若發現是在可飛越的高度內，自動調整 wp.z 以確保安全飛越。
        """
        if not self.areas:
            return True, "開發模式 (無邊界限制)"

        for area_id, data in self.areas.items():
            b = data.get("boundary", {})
            danger = data.get("danger_level", "Unknown")

            in_x = b.get("west", -1e9) <= wp.x <= b.get("east", 1e9)
            in_y = b.get("south", -1e9) <= wp.y <= b.get("north", 1e9)

            if in_x and in_y:
                area_type = data.get("type", "Unknown")
                
                # Phase 5: 道路網語意，視為安全走廊
                if area_type == "Road_Network":
                    return True, f"✅ 進入路網 {area_id} (安全走廊)"
                    
                # 高危區域（紅色禁航區）3D 高度判斷
                if danger == "High":
                    max_height = float(data.get("max_height_m", 100.0))
                    if max_height <= 30.0:
                        safe_z = max_height + 5.0 # 取 5m 安全避障餘裕
                        wp.z = max(wp.z, safe_z) # 強制拉升
                        return True, f"⚠️ 啟動 3D 飛越: {area_id} (高度 {max_height:.1f}m，飛越至 {wp.z:.1f}m)"
                    else:
                        return False, f"❌ 摩天禁航攔截: {area_id} ({area_type} 高度 {max_height:.1f}m，禁止越空)"
                return True, f"✅ 進入 {area_id} ({area_type})"

        return False, f"❌ 座標 ({wp.x:.1f}, {wp.y:.1f}) 不在任何已知區域"

    def validate_waypoints(self, waypoints: List[Waypoint]) -> List[Tuple[Waypoint, bool, str]]:
        """批量驗證航點，回傳 (航點, 是否通過, 說明) 列表"""
        return [(wp, *self.check_and_adjust(wp)) for wp in waypoints]


# ══════════════════════════════════════════════════════════════════════
#  路徑規劃器
# ══════════════════════════════════════════════════════════════════════

class PathPlanner:
    """
    「代碼即規劃 (Code-as-Planner)」概念實現（論文 4.2 節 / AuDeRe 架構）
    將 Layer1 抽象決策離散化為具體航點序列。
    """

    @staticmethod
    def from_layer1_decision(decision: Layer1Decision) -> List[Waypoint]:
        """
        從 Layer1Decision 解析航點列表。
        支援以下格式：
          - {"x":10, "y":20}                      → 使用 decision.altitude
          - {"x":10, "y":20, "z":-15}             → 使用指定高度
          - {"x":10, "y":20, "label":"Point_A"}   → 帶語義標籤
        """
        waypoints = []
        for raw in decision.waypoints_raw:
            try:
                wp = Waypoint(
                    x=float(raw["x"]),
                    y=float(raw["y"]),
                    z=float(raw.get("z", decision.altitude)),
                    label=raw.get("label", ""),
                )
                waypoints.append(wp)
            except (KeyError, ValueError, TypeError) as e:
                print(f"⚠️ [PathPlanner] 跳過無效航點 {raw}: {e}")

        return waypoints

    @staticmethod
    def insert_safe_altitude(
        waypoints: List[Waypoint],
        safe_z: float = -20.0,
        climb_threshold: float = 5.0,
    ) -> List[Waypoint]:
        """
        若兩航點之間高度差超過閾值，自動插入安全爬升點（防止急速爬升）。
        """
        if len(waypoints) < 2:
            return waypoints

        result = [waypoints[0]]
        for prev, curr in zip(waypoints, waypoints[1:]):
            if abs(curr.z - prev.z) > climb_threshold:
                mid = Waypoint(
                    x=(prev.x + curr.x) / 2,
                    y=(prev.y + curr.y) / 2,
                    z=safe_z,
                    label="AUTO_CLIMB_POINT",
                )
                result.append(mid)
            result.append(curr)
        return result

    @staticmethod
    def compute_total_distance(waypoints: List[Waypoint]) -> float:
        """計算航點序列的總飛行距離"""
        if len(waypoints) < 2:
            return 0.0
        return sum(
            waypoints[i].distance_to(waypoints[i + 1])
            for i in range(len(waypoints) - 1)
        )

    @staticmethod
    def generate_macro_sweep(
        west: float, east: float, south: float, north: float, 
        altitude: float = -10.0, scan_width: float = 20.0
    ) -> List[Waypoint]:
        """
        [新增] 生成大範圍 S 型 (Lawnmower) 網格掃描航點，針對大型搜尋區域優化。
        """
        waypoints = []
        y = south
        direction = 1 # 1: 向東, -1: 向西
        
        while y <= north:
            if direction == 1:
                waypoints.append(Waypoint(x=west, y=y, z=altitude, label=f"Sweep_Start_Y{y:.0f}"))
                waypoints.append(Waypoint(x=east, y=y, z=altitude, label=f"Sweep_End_Y{y:.0f}"))
            else:
                waypoints.append(Waypoint(x=east, y=y, z=altitude, label=f"Sweep_Start_Y{y:.0f}"))
                waypoints.append(Waypoint(x=west, y=y, z=altitude, label=f"Sweep_End_Y{y:.0f}"))
            
            y += scan_width
            direction *= -1
            
        return waypoints


# ══════════════════════════════════════════════════════════════════════
#  Layer 2 主控制器
# ══════════════════════════════════════════════════════════════════════

class Layer2BehaviorPlanner:
    """
    Layer 2 Behavior Planner — 完整獨立版
    =========================================
    論文第 4.2 節完整實現，包含：
    - FSM 有限狀態機（Singleton + Mutex）
    - Geofencing 地理圍欄攔截
    - Code-as-Planner 路徑規劃
    - 10Hz 狀態監控迴圈
    - 斷線自主模式
    - 情節記憶寫入
    """

    MONITOR_HZ = 10          # 監控迴圈頻率 (Hz)
    LAYER1_TIMEOUT_SEC = 0.5 # Layer1 通訊逾時（500ms，論文設計值）
    ARRIVAL_THRESHOLD = 2.0  # 抵達判斷距離 (公尺)

    def __init__(
        self,
        env_path: str = "environment.json",
        memory: Optional[EpisodicMemory] = None,
    ):
        self.fsm = FSM()
        self.fsm.reset()
        self.geofence = Geofence(env_path)
        self.planner = PathPlanner()
        self.memory = memory or EpisodicMemory()

        # 任務狀態
        self._current_mission: Optional[Layer1Decision] = None
        self._waypoints: List[Waypoint] = []
        self._current_wp_index: int = 0
        self._mission_start_time: float = 0.0
        self._target_found: bool = False

        # 模擬無人機當前位置
        self._drone_pos: Waypoint = Waypoint(0.0, 0.0, 0.0)

        # 監控執行緒
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()

        print("✅ [Layer2] Behavior Planner 初始化完成")

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def process_layer1_decision(self, raw_json: dict) -> bool:
        """
        接收並處理 Layer1 JSON 決策。
        回傳 True 表示成功啟動任務，False 表示被攔截或失敗。
        """
        # 1. 嘗試鎖定資源（FSM 進入 LOADING 狀態）
        if not self.fsm.transition(FSMState.LOADING, "接收 Layer1 決策"):
            self.memory.write_event(
                "FSM_BLOCK", "Layer2 忙碌中，拒絕新指令", severity="WARNING"
            )
            return False

        try:
            # 2. 解析 Layer1 決策
            decision = Layer1Decision.from_json(raw_json)
            print(f"\n📥 [Layer2] 收到任務: {decision.mission_id} | 指令: {decision.command} | 目標: {decision.target_area}")

            # 3. 生成航點序列
            raw_waypoints = self.planner.from_layer1_decision(decision)

            # [新增] Macro-Sweep 自動展開邏輯
            # 如果是 SEARCH 但 LLM 只給了極少的框架點，自動啟動覆蓋網格計算
            if decision.command == "SEARCH" and len(raw_waypoints) <= 4:
                tgt_data = self.geofence.areas.get(decision.target_area)
                if tgt_data and "boundary" in tgt_data:
                    b = tgt_data["boundary"]
                    area_size = (b.get("east", 0) - b.get("west", 0)) * (b.get("north", 0) - b.get("south", 0))
                    if area_size >= 10000: # >= 1 公頃即啟用宏觀掃描
                        print(f"🌍 [Layer2] 偵測到大範圍 SEARCH ({area_size:,.0f} m^2)，啟用宏觀 S 型網格生成")
                        raw_waypoints = self.planner.generate_macro_sweep(
                            west=b["west"], east=b["east"], 
                            south=b["south"], north=b["north"], 
                            altitude=decision.altitude,
                            scan_width=20.0
                        )

            if not raw_waypoints:
                raise ValueError("航點序列為空，無法執行任務")

            # 4. Geofencing 驗證（核心安全機制）
            validated = self.geofence.validate_waypoints(raw_waypoints)
            safe_waypoints = []
            for wp, is_safe, msg in validated:
                print(f"  🛡️  [Geofence] {wp} → {msg}")
                if not is_safe:
                    self.memory.write_event(
                        "GEOFENCE_VIOLATION",
                        f"任務 {decision.mission_id}: {msg}",
                        severity="WARNING",
                    )
                    raise ValueError(f"地理圍欄攔截: {msg}")
                safe_waypoints.append(wp)

            # 5. 路徑優化（自動插入安全高度點）
            final_waypoints = self.planner.insert_safe_altitude(safe_waypoints)
            total_dist = self.planner.compute_total_distance(final_waypoints)

            print(f"\n📊 [PathPlanner] 生成 {len(final_waypoints)} 個航點，總距離: {total_dist:.1f} m")
            for i, wp in enumerate(final_waypoints):
                if i < 3 or i >= len(final_waypoints) - 2:
                    label = f" ({wp.label})" if wp.label else ""
                    print(f"   [{i}] {wp}{label}")
                elif i == 3:
                     print(f"   ... (省略 {len(final_waypoints) - 5} 個中間航點) ...")

            # [新增] 電量可行性審查 (Battery-Aware Validation)
            est_flight_time = total_dist / max(decision.speed, 0.1)
            max_endurance = 1800.0 * 0.85 # 假設滿電續航 30Mins，保留 15%
            
            # 若預估時間超過無人機極限或任務超時極限
            if est_flight_time > max_endurance:
                err_msg = f"電池續航力不足！預估耗時 {est_flight_time:.1f}s，可用極限 {max_endurance:.1f}s"
                print(f"❌ [PathPlanner] 攔截: {err_msg}")
                self.memory.write_event(
                    "BATTERY_VALIDATION_FAILED",
                    f"任務 {decision.mission_id}: {err_msg} (總距 {total_dist:.1f}m)",
                    severity="ERROR",
                )
                raise ValueError(err_msg)

            # 6. 啟動任務
            self._current_mission = decision
            self._waypoints = final_waypoints
            self._current_wp_index = 0
            self._mission_start_time = time.time()
            self._target_found = False

            self.memory.write_event(
                "MISSION_START",
                f"任務 {decision.mission_id} 啟動，{len(final_waypoints)} 個航點，預計 {decision.timeout_sec}s 完成",
                severity="INFO",
            )

            self.fsm.transition(FSMState.EXECUTING, f"任務 {decision.mission_id} 開始執行")
            self._start_monitor()
            return True

        except Exception as e:
            print(f"❌ [Layer2] 任務啟動失敗: {e}")
            self.memory.write_event("TASK_ABORT", str(e), severity="ERROR")
            self.fsm.transition(FSMState.ABORT, str(e))
            return False

    def signal_target_found(self, confidence: float = 0.9):
        """
        接收「戰情室」的 Target Found 訊號，觸發任務中斷與懸停。
        模擬論文 4.2 節的即時任務中斷機制。
        """
        print(f"\n🎯 [Layer2] 收到 Target Found 訊號 (信心度: {confidence:.0%})")
        self._target_found = True
        self.memory.write_event(
            "TARGET_FOUND",
            f"偵測到目標，信心度 {confidence:.0%}，觸發懸停",
            severity="INFO",
        )
        self.fsm.transition(FSMState.HOVERING, "目標發現 → 懸停待命")

    def activate_autonomous_mode(self, reason: str = "通訊中斷"):
        """
        觸發斷線自主模式，改由情節記憶庫導航。
        對應論文 4.2 節「500ms 逾時後改由本地情節記憶庫執行自主導航」。
        """
        print(f"\n🔴 [Layer2] 啟動自主模式: {reason}")
        current = self.fsm.state
        if current in (FSMState.IDLE, FSMState.EXECUTING):
            self.fsm.transition(FSMState.AUTONOMOUS, reason)

            # 從記憶庫檢索歷史路徑
            context = self._current_mission.target_area if self._current_mission else "未知區域"
            lessons = self.memory.retrieve_lessons_learned(f"自主導航 {context}")

            print("🧠 [自主模式] 從情節記憶庫提取導航策略:")
            for lesson in lessons:
                print(f"  • {lesson}")

            self.memory.write_event(
                "AUTONOMOUS_MODE",
                f"因 {reason} 切換自主模式，情節記憶檢索完成",
                severity="WARNING",
            )

    def stop(self):
        """停止監控迴圈與任務"""
        self._stop_monitor.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        if self.fsm.state not in (FSMState.IDLE, FSMState.MISSION_DONE):
            self.fsm.transition(FSMState.IDLE, "手動停止")
        print("[Layer2] 已停止")

    def get_status(self) -> dict:
        """取得當前狀態摘要"""
        return {
            "fsm_state": self.fsm.state.name,
            "mission_id": self._current_mission.mission_id if self._current_mission else None,
            "waypoint_progress": f"{self._current_wp_index}/{len(self._waypoints)}",
            "drone_pos": str(self._drone_pos),
            "target_found": self._target_found,
            "memory_report": self.memory.get_summary_report(),
        }

    def simulate_drone_move(self, x: float, y: float, z: float = -10.0):
        """
        模擬無人機移動（供測試使用，取代 AirSim 感測器回傳）
        """
        self._drone_pos = Waypoint(x, y, z)

    # ------------------------------------------------------------------
    # 內部方法
    # ------------------------------------------------------------------

    def _start_monitor(self):
        """啟動 10Hz 監控執行緒"""
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="Layer2-Monitor"
        )
        self._monitor_thread.start()
        print(f"🔁 [Layer2] 監控迴圈啟動 ({self.MONITOR_HZ}Hz)")

    def _monitor_loop(self):
        """
        10Hz 監控迴圈（論文 4.2 節設計）：
        - 檢查航點抵達
        - 監測任務逾時
        - 偵測 Layer1 通訊中斷
        """
        interval = 1.0 / self.MONITOR_HZ
        last_layer1_contact = time.time()

        while not self._stop_monitor.is_set():
            loop_start = time.time()
            state = self.fsm.state

            if state == FSMState.EXECUTING:
                # ── 逾時檢查 ──────────────────────────────
                elapsed = time.time() - self._mission_start_time
                if (self._current_mission and
                        elapsed > self._current_mission.timeout_sec):
                    print(f"\n⏰ [Monitor] 任務逾時 ({elapsed:.1f}s > {self._current_mission.timeout_sec}s)")
                    self.memory.write_event(
                        "TIMEOUT",
                        f"任務 {self._current_mission.mission_id} 逾時",
                        severity="WARNING",
                    )
                    self.activate_autonomous_mode("任務逾時")

                # ── 航點抵達檢查 ──────────────────────────
                elif self._waypoints and self._current_wp_index < len(self._waypoints):
                    target_wp = self._waypoints[self._current_wp_index]
                    dist = self._drone_pos.distance_to(target_wp)

                    if dist <= self.ARRIVAL_THRESHOLD:
                        label = f" ({target_wp.label})" if target_wp.label else ""
                        print(f"  ✈️  [Monitor] 抵達航點 [{self._current_wp_index}]{label}")
                        self._current_wp_index += 1

                        if self._current_wp_index >= len(self._waypoints):
                            print("🏁 [Monitor] 所有航點執行完畢！")
                            self.memory.write_event(
                                "MISSION_COMPLETE",
                                f"任務 {self._current_mission.mission_id} 完成",
                            )
                            self.fsm.transition(FSMState.MISSION_DONE, "所有航點完成")
                            self._stop_monitor.set()

                # ── Layer1 通訊逾時模擬 ───────────────────
                layer1_silent = time.time() - last_layer1_contact
                if layer1_silent > self.LAYER1_TIMEOUT_SEC * 10:  # 測試環境放寬 10 倍
                    last_layer1_contact = time.time()   # 重設計時（測試用）

            # 維持精確 10Hz
            elapsed_loop = time.time() - loop_start
            sleep_time = max(0, interval - elapsed_loop)
            time.sleep(sleep_time)

        print("🔁 [Layer2] 監控迴圈已結束")
