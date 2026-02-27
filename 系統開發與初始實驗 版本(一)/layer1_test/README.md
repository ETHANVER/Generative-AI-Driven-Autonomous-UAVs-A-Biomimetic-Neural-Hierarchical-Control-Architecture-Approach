# Layer 1 Brain — 獨立測試套件

> 根據論文《基於仿生神經分工架構之生成式 AI 自主無人機研究》第 4.1 節實作

## 概述

本資料夾為 **Layer 1 (LLM Brain + RAG)** 的**完全獨立測試環境**，  
**無需安裝 OpenAI API Key、ChromaDB 或 AirSim**，直接在純 Python 環境執行。

## 檔案結構

```
layer1_test/
├── db/                      # 四大 RAG 資料庫
│   ├── environment_db.json  # 語意網格地圖（19 個語意分塊）
│   ├── rules_db.json        # 飛行 SOP 與安全規則（11 個語意分塊）
│   ├── conditions_db.json   # 硬體限制與觸發條件（10 個語意分塊）
│   └── targets_db.json      # 偵蒐目標特徵模板（8 個語意分塊）
├── mock_rag.py              # TF-IDF RAG 引擎 (取代 ChromaDB)
├── mock_llm.py              # 規則式 LLM 模擬器 (取代 OpenAI GPT-4o)
├── layer1_standalone.py     # Layer 1 Brain 完整整合器
├── test_layer1.py           # 自動化測試套件（16 個測試案例）
├── run_demo.py              # 互動式 4 場景 Demo
└── README.md                # 本說明文件
```

## 快速開始

### 1. 執行自動化測試 (Standalone)
驗證 RAG 邏輯與 MockLLM 決策流程：
```bash
python test_layer1.py
```

### 2. 執行交互式 Demo (Standalone)
模擬與大腦的自然語言對話：
```bash
python run_demo.py
```

### 3. 執行 AirSim 整合實驗 (AirSim Required)
**[NEW]** 從實體模擬器獲取座標進行決策：
```bash
python run_layer1_airsim.py
```

## 免費組件配置說明

本資料夾預設已全面切換為**免付費開源組件**，以符合預算與商品化需求：
- **LLM**: 預設使用 `Ollama (llava:13b)`。請確保已安裝 Ollama 伺服器，或本腳本會自動降級至本地 `MockLLM`。
- **Embedding**: 使用 `sentence-transformers/all-MiniLM-L6-v2`，無需 OpenAI Key。
- **向量資料庫**: 原生使用 `ChromaDB` 持久化儲存 HNSW 索引。

## 論文精準對齊節點

- **相似度閾值**: 嚴格執行 `0.75` 語意過濾。
- **自適應降級**: 若 0.75 範圍內無命中，自動降至 0.50 重試。
- **情節記憶**: 具備「負面經驗」權重加成，攔截過去失敗的決策路徑。

## 四大 RAG 資料庫說明

| 資料庫 | 檔案 | 分塊數 | 用途 |
|--------|------|--------|------|
| 環境資料庫 | `environment_db.json` | 19 | 語意網格地圖：區域類型、邊界、高度建議 |
| 規則資料庫 | `rules_db.json` | 11 | 飛行 SOP：高度限制、禁航、幻覺攔截、電量 |
| 條件資料庫 | `conditions_db.json` | 10 | 硬體條件：通訊逾時、速限、APF 範圍、IMU |
| 目標資料庫 | `targets_db.json` | 8  | 偵蒐目標：人員、車輛、設施、未知目標 |

## 測試案例說明

| 編號 | 測試內容 | 對應論文機制 |
|------|----------|-------------|
| TC01 | 四大資料庫均載入成功 | 資料庫建置 |
| TC02-05 | 各資料庫分塊數量驗證 | Semantic Chunking |
| TC06 | 相似度閾值 0.75 過濾 | Top-K Filter |
| TC07 | Top-K=3 數量上限 | Top-K Retrieval |
| TC08 | 情境感知混合查詢 | Context-Aware Retrieval |
| TC09 | 森林區規則命中 | RAG Precision |
| TC10 | 禁航區約束命中 | RAG Precision |
| TC11 | [Constraints] 區塊格式 | Dynamic Prompt Injection |
| TC12 | JSON 輸出格式驗證 | LLM Output Schema |
| TC13a-e | 任務/區域類型解析 | NLU / Task Parsing |
| TC14 | 端到端完整管道 | Full Pipeline |
| TC15 | 低電量強制 RTH | Safety Override |
| TC16 | Layer 2 Schema 相容性 | Cross-layer Interface |

## Demo 場景說明

| 場景 | 主題 |
|------|------|
| A | 完整 4 步驟 RAG 管道（自然語言 → JSON 決策）|
| B | RAG 效果對比（有/無 Constraints 差異）|
| C | 電量安全機制（80%/35%/19% 三種狀態）|
| D | 閉環學習（Layer 2 Target Found 回報影響決策）|

## 切換至真實 OpenAI GPT-4o

只需修改 `layer1_standalone.py` 中的初始化參數：

```python
# 由 Mock 模式 ↓
brain = Layer1Brain(db_dir="db", use_openai=False)

# 切換為真實 GPT-4o ↓
brain = Layer1Brain(
    db_dir="db",
    use_openai=True,
    openai_api_key="sk-your-api-key-here"
)
```

## 切換至真實 ChromaDB

修改 `layer1_standalone.py` 中的 RAGEngine 初始化，  
改用正式 `chromadb` + `openai` `text-embedding-3-small` 即可。

## Layer 1 完整架構

```
自然語言任務指令
    │
    ▼
┌──────────────────────────────────────────────┐
│              Layer 1 Brain                   │
│                                              │
│  ① 語意分塊向量化 (TF-IDF / text-embedding)  │
│     ↓ 四大 RAG 資料庫建立索引                │
│                                              │
│  ② 情境感知檢索 (Context-Aware Retrieval)    │
│     混合「當前狀態」+「任務指令」查詢向量     │
│                                              │
│  ③ Top-K=3 + 閾值 0.75 語意過濾             │
│     過濾低相關語意雜訊                       │
│                                              │
│  ④ 動態 Prompt 注入 [Constraints]            │
│     填入 System Prompt 約束區塊              │
│                                              │
│  ⑤ LLM 推論 (MockLLM / GPT-4o)              │
│     Chain-of-Thought + JSON 輸出             │
└──────────────────────────────────────────────┘
    │
    ▼
Layer 2 JSON 決策
(mission_id, command, waypoints, ...)
```
