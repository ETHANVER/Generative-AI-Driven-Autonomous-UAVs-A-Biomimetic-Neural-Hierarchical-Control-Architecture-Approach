"""
mock_llm.py - 規則式 LLM 模擬器（不依賴 OpenAI API）
======================================================
模擬 GPT-4o 的行為：
  - 接收含 RAG [Constraints] 區塊的 System Prompt
  - 接收自然語言任務指令 (User Prompt)
  - 輸出符合 Layer 2 JSON Schema 的標準決策

支援 4 種任務模式（PATROL / RECON / RTH / SEARCH），
以及一鍵切換為真實 OpenAI 客戶端。

切換方式（使用真實 GPT-4o）：
  from mock_llm import get_llm_client
  client = get_llm_client(use_openai=True, api_key="sk-...")
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# Layer 2 標準 JSON Schema
# ══════════════════════════════════════════════════════════════════════

def build_decision_json(
    command: str,
    target_area: str,
    waypoints: list,
    altitude: float = -10.0,
    speed: float = 5.0,
    priority: int = 5,
    timeout_sec: float = 60.0,
    reasoning: str = "",
    constraints_applied: list = None,
) -> dict:
    """建立符合 Layer 2 標準 JSON Schema 的決策物件"""
    return {
        "mission_id":           f"L1_{command}_{uuid.uuid4().hex[:6].upper()}",
        "command":              command,
        "target_area":          target_area,
        "waypoints":            waypoints,
        "altitude":             altitude,
        "speed":                speed,
        "priority":             priority,
        "timeout_sec":          timeout_sec,
        "reasoning":            reasoning,
        "constraints_applied":  constraints_applied or [],
        "generated_by":         "MockLLM",
        "timestamp":            time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ══════════════════════════════════════════════════════════════════════
# 規則式任務解析器
# ══════════════════════════════════════════════════════════════════════

class TaskParser:
    """從自然語言指令解析任務類型與目標區域"""

    COMMAND_KEYWORDS = {
        "PATROL":  ["巡邏", "patrol", "巡查", "偵察巡飛", "monitor", "掃描"],
        "RECON":   ["偵蒐", "recon", "reconnaissance", "偵察", "搜尋", "search", "偵查"],
        "SEARCH":  ["搜索", "search", "找到", "尋找", "find", "locate", "目標搜索"],
        "RTH":     ["返航", "return", "rth", "回家", "返回", "home"],
        "HOVER":   ["懸停", "hover", "停留", "wait", "等待"],
    }

    AREA_KEYWORDS = {
        "Area1": ["area1", "area 1", "森林", "forest", "area1"],
        "Area2": ["area2", "area 2", "建築", "building", "city", "城市"],
        "Area3": ["area3", "area 3", "開闊", "open", "平原", "field"],
    }

    @classmethod
    def parse_command(cls, instruction: str) -> str:
        lowered = instruction.lower()
        for cmd, keywords in cls.COMMAND_KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                return cmd
        return "PATROL"  # 預設

    @classmethod
    def parse_area(cls, instruction: str) -> str:
        lowered = instruction.lower()
        for area, keywords in cls.AREA_KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                return area
        return "Area1"  # 預設


# ══════════════════════════════════════════════════════════════════════
# 航點產生器（依區域與任務類型）
# ══════════════════════════════════════════════════════════════════════

AREA_WAYPOINT_PLANS = {
    "Area1": {
        "PATROL": [
            {"x": 10, "y": 10, "z": -15, "label": "ForestEntry"},
            {"x": 30, "y": 20, "z": -15, "label": "PatrolAlpha"},
            {"x": 50, "y": 50, "z": -12, "label": "PatrolBeta"},
            {"x": 70, "y": 70, "z": -12, "label": "PatrolGamma"},
            {"x": 80, "y": 30, "z": -15, "label": "PatrolDelta"},
            {"x": 50, "y": 10, "z": -15, "label": "HoverPoint"},
        ],
        "RECON": [
            {"x": 20, "y": 20, "z": -18, "label": "ReconStart"},
            {"x": 40, "y": 60, "z": -18, "label": "ReconSweep1"},
            {"x": 80, "y": 50, "z": -18, "label": "ReconSweep2"},
            {"x": 60, "y": 20, "z": -18, "label": "ReconSweep3"},
        ],
        "SEARCH": [
            {"x": 15, "y": 15, "z": -10, "label": "SearchGrid1"},
            {"x": 45, "y": 15, "z": -10, "label": "SearchGrid2"},
            {"x": 75, "y": 15, "z": -10, "label": "SearchGrid3"},
            {"x": 75, "y": 50, "z": -10, "label": "SearchGrid4"},
            {"x": 45, "y": 50, "z": -10, "label": "SearchGrid5"},
            {"x": 15, "y": 80, "z": -10, "label": "SearchGrid6"},
        ],
        "RTH": [{"x": 0, "y": 0, "z": -5, "label": "HomeBase"}],
        "HOVER": [{"x": 50, "y": 50, "z": -15, "label": "HoverCenter"}],
    },
    "Area2": {
        "PATROL": [
            {"x": 105, "y": 105, "z": -55, "label": "BypassEntry"},
            {"x": 110, "y": 120, "z": -55, "label": "BypassAlpha"},
        ],
        "RECON": [
            {"x": 105, "y": 110, "z": -60, "label": "Area2Recon"},
        ],
        "SEARCH": [{"x": 105, "y": 105, "z": -55, "label": "Area2Search"}],
        "RTH":    [{"x": 0, "y": 0, "z": -5, "label": "HomeBase"}],
        "HOVER":  [{"x": 105, "y": 105, "z": -55, "label": "Area2Hover"}],
    },
    "Area3": {
        "PATROL": [
            {"x": 10,  "y": -10,  "z": -8,  "label": "OpenFieldEntry"},
            {"x": 50,  "y": -30,  "z": -8,  "label": "SweepAlpha"},
            {"x": 90,  "y": -60,  "z": -8,  "label": "SweepBeta"},
            {"x": 50,  "y": -90,  "z": -8,  "label": "SweepGamma"},
            {"x": 10,  "y": -80,  "z": -8,  "label": "SweepDelta"},
        ],
        "RECON": [
            {"x": 20, "y": -20, "z": -15, "label": "Area3Recon1"},
            {"x": 80, "y": -50, "z": -15, "label": "Area3Recon2"},
            {"x": 50, "y": -80, "z": -15, "label": "Area3Recon3"},
        ],
        "SEARCH": [
            {"x": 25, "y": -25, "z": -5,  "label": "LowAltSearch1"},
            {"x": 75, "y": -25, "z": -5,  "label": "LowAltSearch2"},
            {"x": 75, "y": -75, "z": -5,  "label": "LowAltSearch3"},
            {"x": 25, "y": -75, "z": -5,  "label": "LowAltSearch4"},
        ],
        "RTH":   [{"x": 0, "y": 0, "z": -5, "label": "HomeBase"}],
        "HOVER": [{"x": 50, "y": -50, "z": -8, "label": "OpenFieldHover"}],
    },
}

AREA_CONFIG = {
    "Area1": {"altitude": -15.0, "speed": 4.0, "timeout": 90.0},
    "Area2": {"altitude": -55.0, "speed": 2.0, "timeout": 60.0},
    "Area3": {"altitude": -8.0,  "speed": 5.0, "timeout": 120.0},
}


# ══════════════════════════════════════════════════════════════════════
# Mock LLM 核心推論器
# ══════════════════════════════════════════════════════════════════════

class MockLLM:
    """
    規則式 GPT-4o 模擬器。
    接收完整 Prompt（含 RAG Constraints 區塊），輸出任務 JSON 決策。
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._call_count = 0

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """
        模擬 openai.chat.completions.create() 的回傳值（JSON 字串）。
        Args:
            system_prompt: 包含 RAG [Constraints] 的 System Prompt
            user_prompt:   自然語言任務指令
        Returns:
            JSON 字串（完全對應 Layer 2 Decision Schema）
        """
        self._call_count += 1

        if self.verbose:
            print(f"\n{'─'*60}")
            print(f"🤖 [MockLLM #{self._call_count}] 收到推論請求")
            print(f"   User: {user_prompt[:100]}")

        # ── 解析任務指令 ──────────────────────────────────
        command = TaskParser.parse_command(user_prompt)
        area    = TaskParser.parse_area(user_prompt)

        # ── 讀取 RAG 約束 ──────────────────────────────────
        constraints_used = self._extract_constraints(system_prompt)

        # ── 讀取情節記憶（閉環學習模擬） ─────────────────────────
        forbidden_zones = self._extract_forbidden_zones(system_prompt)
        if area in forbidden_zones:
            if self.verbose:
                print(f"⚠️  [MockLLM] 偵測到歷史失敗，強制轉為 RTH 避開 {area}")
            command = "RTH"
            area = "Home"
            reasoning = f"偵測到 {forbidden_zones[0]} 區域在過去任務中曾遭攔截，為確保安全，本次任務改為返航。"
        area_plans = AREA_WAYPOINT_PLANS.get(area, AREA_WAYPOINT_PLANS["Area1"])
        waypoints  = area_plans.get(command, area_plans.get("PATROL", []))

        # ── 讀取區域配置 ───────────────────────────────────
        cfg = AREA_CONFIG.get(area, {"altitude": -10.0, "speed": 5.0, "timeout": 60.0})

        # ── 生成推論說明（Chain-of-Thought 模擬）──────────
        reasoning = (
            f"根據任務指令「{user_prompt[:50]}」，識別指令類型為 {command}，"
            f"目標區域為 {area}。"
            f"RAG 約束條件已套用 {len(constraints_used)} 條規則，"
            f"生成符合飛行 SOP 的 {len(waypoints)} 個航點序列。"
        )

        decision = build_decision_json(
            command=command,
            target_area=area,
            waypoints=waypoints,
            altitude=cfg["altitude"],
            speed=cfg["speed"],
            timeout_sec=cfg["timeout"],
            priority=5,
            reasoning=reasoning,
            constraints_applied=constraints_used,
        )

        output_json = json.dumps(decision, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"   → 指令:{command} | 區域:{area} | 航點:{len(waypoints)} 個 | 約束:{len(constraints_used)} 條")

        return output_json

    def _extract_constraints(self, system_prompt: str) -> list:
        """從 System Prompt 中提取已套用的約束條目"""
        lines = system_prompt.split("\n")
        applied = []
        for line in lines:
            if line.strip().startswith("•"):
                # 提取規則 ID 如果有
                applied.append(line.strip()[1:].strip())
        return applied

    def _extract_forbidden_zones(self, system_prompt: str) -> list:
        """從 History 區塊提取發生過攔截的區域"""
        forbidden = []
        if "[History]" in system_prompt:
            # 尋找帶有 ⚠️ 或 Negative/Blocked 標記的行
            lines = system_prompt.split("\n")
            in_history = False
            for line in lines:
                if "[History]" in line:
                    in_history = True
                    continue
                if in_history and not line.strip():
                    in_history = False
                    continue
                
                if in_history and ("⚠️" in line or "Negative" in line or "Blocked" in line):
                    # 嘗試提取區域名稱 (e.g. 區域:Area2)
                    match = re.search(r"區域:(\w+)", line)
                    if match:
                        forbidden.append(match.group(1))
        return forbidden

    @property
    def call_count(self) -> int:
        return self._call_count



# ══════════════════════════════════════════════════════════════════════
# OllamaLLM — 免費本地 LLM（替代 GPT-4o）
# ══════════════════════════════════════════════════════════════════════

class OllamaLLM:
    """
    Ollama 本地 LLM 客戶端（免費替代 GPT-4o）。
    支援任何已安裝的 Ollama 模型（llava:13b, llama3.2:3b 等）。

    論文等效性：
      GPT-4o → Ollama + llava:13b（本地，免費，支援 JSON 輸出）
    """

    # 自動偵測模型優先序
    MODEL_PRIORITY = [
        "llama3.1:8b", "llama3.1:latest",
        "llama3.2:3b", "llama3.2:latest",
        "llama3:latest", "llama3:8b",
        "qwen2.5:3b",  "qwen2.5:7b",
        "llava:13b",   "llava:latest",
        "mistral:latest", "gemma2:latest",
    ]

    JSON_EXTRACT_PROMPT = (
        "\n\n重要：你的輸出必須是且只能是一個 JSON 物件"
        "（不含 markdown code fence ``` 或任何其他文字）。"
        "直接輸出 { 開頭的 JSON。"
    )

    def __init__(self, model: str = "", verbose: bool = True, timeout: int = 60):
        """
        Args:
            model:   Ollama 模型名稱，留空則自動偵測已安裝模型
            verbose: 是否輸出詳細日誌
            timeout: 請求超時秒數
        """
        self.verbose  = verbose
        self.timeout  = timeout
        self._call_count = 0
        self.model = model or self._detect_model()
        print(f"  [OllamaLLM] 使用模型: {self.model}")

    def _detect_model(self) -> str:
        """自動偵測已安裝的 Ollama 模型"""
        try:
            import ollama
            installed = {m.model for m in ollama.list().models}
            for preferred in self.MODEL_PRIORITY:
                if preferred in installed:
                    return preferred
            # 使用第一個可用的
            if installed:
                return next(iter(installed))
        except Exception as e:
            print(f"  [OllamaLLM] 模型偵測失敗: {e}")
        return "llava:13b"  # 使用者已安裝的模型

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """
        呼叫 Ollama 本地 LLM，輸出 JSON 字串。
        若 Ollama 無回應或解析失敗，自動降級到 MockLLM。
        """
        self._call_count += 1
        if self.verbose:
            print(f"\n{'─'*60}")
            print(f"🦙 [OllamaLLM #{self._call_count}] 模型={self.model}")
            print(f"   User: {user_prompt[:80]}")

        full_system = system_prompt + self.JSON_EXTRACT_PROMPT

        try:
            import ollama
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user",   "content": user_prompt},
                ],
                options={"temperature": temperature},
            )
            raw = resp.message.content.strip()

            if self.verbose:
                print(f"   → 原始輸出長度: {len(raw)} chars")

            # 嘗試提取 JSON
            return self._extract_json(raw, user_prompt, system_prompt)

        except Exception as e:
            print(f"  ⚠️ [OllamaLLM] 呼叫失敗 ({e})，降級 MockLLM")
            return MockLLM(verbose=False).chat_completion(system_prompt, user_prompt)

    def _extract_json(self, raw: str, user_prompt: str, system_prompt: str) -> str:
        """從模型輸出中提取合法 JSON"""
        # 嘗試直接解析
        try:
            json.loads(raw)
            return raw
        except json.JSONDecodeError:
            pass

        # 去除 markdown code fence
        cleaned = re.sub(r'```(?:json)?\n?', '', raw).strip()
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            pass

        # 提取 {...} 區塊
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                candidate = match.group()
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # 完全解析失敗 → MockLLM fallback
        print("  ⚠️ [OllamaLLM] JSON 解析失敗，降級 MockLLM")
        return MockLLM(verbose=False).chat_completion(system_prompt, user_prompt)

    @property
    def call_count(self) -> int:
        return self._call_count


# ══════════════════════════════════════════════════════════════════════
# LLM 工廠（支援 Mock / Ollama / OpenAI）
# ══════════════════════════════════════════════════════════════════════

def get_llm_client(
    use_openai: bool = False,
    use_ollama: bool = False,
    api_key:    str  = "",
    model:      str  = "",
    verbose:    bool = True,
):
    """
    LLM 客戶端工廠。

    優先序：
      use_openai=True → OpenAI GPT-4o（需 api_key，付費）
      use_ollama=True → Ollama 本地 LLM（免費，已安裝即能用）
      預設           → MockLLM（規則式，完全免費，無需網路）
    """
    if use_openai and api_key:
        try:
            from openai import OpenAI

            class OpenAIWrapper:
                def __init__(self, key, mdl):
                    self._client = OpenAI(api_key=key)
                    self._model  = mdl or "gpt-4o"
                    self._call_count = 0

                def chat_completion(self, system_prompt, user_prompt, temperature=0.0):
                    self._call_count += 1
                    resp = self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_prompt},
                        ],
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                    return resp.choices[0].message.content

                @property
                def call_count(self): return self._call_count

            print(f"  [LLM] 使用真實 OpenAI GPT-4o（{model or 'gpt-4o'}）")
            return OpenAIWrapper(api_key, model)

        except ImportError:
            print("⚠️  openai 未安裝，回退 OllamaLLM")
            use_ollama = True

    if use_ollama:
        return OllamaLLM(model=model, verbose=verbose)

    return MockLLM(verbose=verbose)


# ══════════════════════════════════════════════════════════════════════
# 快速測試
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    use_ol = "--ollama" in sys.argv

    llm = get_llm_client(use_ollama=use_ol, verbose=True)
    print(f"\nLLM 類型: {type(llm).__name__}")

    system_prompt = """你是一個無人機任務規劃 AI。
[Constraints]
  • [規則] 森林區高度限制：飛行高度保持 10m-20m
  • [條件] 通訊逾時：超過 500ms 切換自主模式
"""
    result = llm.chat_completion(system_prompt, "前往 Area1 執行森林巡邏任務")
    decision = json.loads(result)
    print(f"\n✅ 決策: {decision['command']} @ {decision['target_area']}, "
          f"{len(decision['waypoints'])} 個航點")


