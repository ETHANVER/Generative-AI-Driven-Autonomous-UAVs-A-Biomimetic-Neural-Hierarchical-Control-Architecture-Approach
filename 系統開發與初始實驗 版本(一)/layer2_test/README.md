# Layer 2 Behavior Planner — 獨立測試套件

> 根據論文《基於仿生神經分工架構之生成式 AI 自主無人機研究》第 4.2 節實作

## 概述

本資料夾為 **Layer 2 (Behavior Planner)** 的**完全獨立測試環境**，  
**無需安裝 AirSim、OpenAI API 或 ChromaDB**，可直接在純 Python 環境執行。

## 檔案結構

```
layer2_test/
├── layer2_standalone.py   # Layer 2 完整實作（FSM + Geofencing + PathPlanner + Monitor）
├── mock_memory.py         # 輕量模擬版 EpisodicMemory（取代 ChromaDB）
├── test_layer2.py         # 自動化測試套件（14 個測試案例）
├── run_demo.py            # 互動式 3 場景 Demo
├── environment.json       # 地理邊界設定（3 個飛行區域）
└── README.md              # 本說明文件
```

## 快速開始

### 系統需求
- Python 3.10+
- 無需額外 pip 安裝（僅使用標準函式庫）

### 1. 執行自動化測試 (Standalone)
驗證有限狀態機 (FSM) 轉換與地理圍欄攔截邏輯：
```bash
cd layer2_test
python test_layer2.py
```

### 2. 執行交互式 Demo (Standalone)
模擬一個完整的巡邏任務，觀察 FSM 狀態跳轉與監控迴圈：
```bash
cd layer2_test
python run_demo.py
```

### 3. 執行 AirSim 整合實驗 (AirSim Required)
**[NEW]** 從實體模擬器同步位置並監控地理邊界：
```bash
cd layer2_test
python run_layer2_airsim.py
```

## 核心機制優化

- **FSM Singleton**: 使用 `RLock` 與 `Singleton` 模式，確保在高頻 (10Hz+) 環境下讀寫狀態的一致性。
- **Autonomous Mode**: 實作了論文 4.2 節提到的「通訊斷線 500ms 自動切換」，此時 Layer 2 會主動從情節記憶庫提取導航策略。
- **Geofencing**: 已整合 `environment.json`，實現對 LLM 幻覺座標的硬性物理隔離。

## 測試案例說明

| 編號 | 測試內容 | 對應論文機制 |
|------|----------|-------------|
| TC01 | FSM 合法狀態轉換 | 有限狀態機 (FSM) |
| TC02 | FSM 非法轉換防護 | Singleton + Mutex |
| TC03 | Geofence 合法航點通過 | 地理圍欄 |
| TC04 | Geofence 禁航區攔截 | LLM 幻覺攔截 |
| TC05 | Geofence 未知座標攔截 | 地理圍欄 |
| TC06 | 完整任務流程 | Layer1→Layer2 整合 |
| TC07 | 航點 JSON 解析 | Code-as-Planner |
| TC08 | 安全高度自動插入 | 飛行安全機制 |
| TC09 | 斷線自主模式觸發 | 情節記憶自主導航 |
| TC10 | Geofence 攔截整體任務 | 幻覺攔截整合 |
| TC11 | 情節記憶寫入與檢索 | 閉環學習機制 |
| TC12 | Target Found 懸停 | 即時任務中斷 |
| TC13 | 空航點序列防護 | 輸入驗證 |
| TC14 | 非法 JSON 輸入防護 | 輸入驗證 |

## Demo 場景說明

| 場景 | 主題 | 展示機制 |
|------|------|----------|
| A | 正常巡邏任務 | FSM 生命週期、路徑規劃、10Hz 監控 |
| B | LLM 幻覺攔截 | Geofencing、情節記憶負面標記 |
| C | 通訊中斷自主飛行 | 斷線偵測、情節記憶自主導航 |

## Layer 2 核心架構

```
Layer1 JSON 決策
      │
      ▼
┌─────────────────────────────────────────┐
│         Layer 2 Behavior Planner        │
│                                         │
│  ① FSM 資源鎖定 (IDLE → LOADING)       │
│  ② Geofencing 驗證 (攔截 LLM 幻覺)     │
│  ③ Code-as-Planner 路徑規劃            │
│  ④ FSM 啟動任務 (LOADING → EXECUTING)  │
│  ⑤ 10Hz 監控迴圈                       │
│     ├── 航點抵達確認                    │
│     ├── 任務逾時偵測                    │
│     └── Layer1 通訊逾時 → 自主模式      │
│  ⑥ 情節記憶寫入 (閉環學習)             │
└─────────────────────────────────────────┘
      │
      ▼
 Layer3 航點指令
 (在 SITL 環境中接入 AirSim API)
```

## 接入 AirSim（下一步）

當 AirSim 環境就緒後，只需修改 `layer2_standalone.py` 中的：

```python
# 替換此模擬方法 ↓
def simulate_drone_move(self, x, y, z):
    self._drone_pos = Waypoint(x, y, z)

# 改為從 AirSim 客戶端讀取實際位置 ↓
def get_drone_pos_from_airsim(self, airsim_client):
    state = airsim_client.getMultirotorState()
    pos = state.kinematics_estimated.position
    self._drone_pos = Waypoint(pos.x_val, pos.y_val, pos.z_val)
```

並將 `EpisodicMemory` 從 `mock_memory` 改為正式的 `memory_db` 即可完成 SITL 接入。
