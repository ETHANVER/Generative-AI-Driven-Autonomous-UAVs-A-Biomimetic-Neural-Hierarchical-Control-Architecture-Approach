# Layer 3 Low-Level Flight Controller — 獨立測試套件

> 根據論文《基於仿生神經分工架構之生成式 AI 自主無人機研究》第 4.3 節實作

## 概述

本資料夾為 **Layer 3 (Low-Level Flight Controller / 仿小腦)** 的**完全獨立測試環境**，  
專注於驗證**人工勢場法 (APF)** 避障算法與 **PID 速度控制**邏輯，無需 AirSim 即可執行物理模擬。

## 檔案結構

```
layer3_test/
├── layer3_standalone.py  # Layer 3 核心實作 (APF Engine + PID + Physics Sim)
├── mock_perception.py    # 模擬感知系統 (回傳附近障礙物座標)
├── test_layer3.py        # 自動化測試套件 (5 個測試案例)
├── run_demo.py           # 避障導航展示 Demo
└── README.md             # 本說明文件
```

## 核心機制

1.  **APF 避障引擎**：
    *   **引力 (Attractive Force)**：拉動無人機向目標點前進。
    *   **斥力 (Repulsive Force)**：當距離障礙物小於 `rho_0` (預設 2.5m) 時產生推力。
2.  **PID 速度平滑**：
    *   將合力向量轉化為平滑的速度指令。
    *   **硬性限制**：速度上限嚴格控制在 5.0 m/s 以內 (符合論文安全性要求)。
3.  **50Hz 控制頻率**：
    *   模擬實體無人機的高頻控制需求，確保避障的即時性。

## 快速開始

### 1. 執行自動化測試 (Standalone)
驗證 APF 避障邏輯、PID 輸出限制以及 50Hz 控制穩定性：
```bash
python test_layer3.py
```

### 2. 執行避障展示 Demo (Standalone)
觀察無人機在純模擬環境下如何自動繞過障礙物：
```bash
python run_demo.py
```

### 3. 執行 AirSim 整合實驗 (AirSim Required)
**[NEW]** 實際驅動 AirSim 模擬器中的無人機執行 50Hz APF 避障：
```bash
python run_layer3_airsim.py
```

## 核心機制優化

- **APF Engine**: 實作了引力與斥力的矢量合力計算，斥力場範圍設定為論文建議的 2.5m。
- **PID Smoothing**: 三軸獨立 PID 確保控制向量不會出現階躍式抖動，保護實體馬達。
- **AirSim Driver**: 整合了 `moveByVelocityAsync`，實現了從小腦到執行器的直接神經傳導模擬。

## 接入全系統（下一步）

在整合環境中，`Layer 3` 將透過 `WarRoom` 共享記憶體獲取 `Layer 2` 的目前的航點，
並將 `Perception` 模組辨識出的深度圖障礙物座標餵入 APF 引擎進行實時避障。
