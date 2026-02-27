"""
target_schema_generator.py - 動態目標特徵生成器
=================================================
根據論文第 4.1 節「目標資料庫（Target DB）」設計：

  「User Input → 雲端 LLM 初步解析 → 提取關鍵特徵（如目標顏色、形狀）
   → 自動填入 Target Schema → 存入 Target DB」

功能：
  1. 接收自然語言任務指令（如「搜索紅色可疑車輛」）
  2. 用 MockLLM / GPT-4o 提取目標特徵
  3. 生成符合 targets_db.json Schema 的條目
  4. 動態 Upsert 至 targets_db.json（不覆蓋既有靜態條目）

使用方式：
  gen = TargetSchemaGenerator(targets_db_path="db/targets_db.json")
  new_chunk = gen.generate_from_instruction("搜索綠衣男性，持可疑包裹")
  gen.upsert_to_db(new_chunk)  # 寫回 targets_db.json
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# 特徵提取器（模擬 LLM 解析）
# ══════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    從自然語言提取目標視覺特徵（模擬 GPT-4o 解析行為）。
    正式版可換為真實 LLM 呼叫。
    """

    # 顏色關鍵字（中英文）
    COLOR_KW = {
        "紅": "red", "紅色": "red", "red": "red",
        "藍": "blue", "藍色": "blue", "blue": "blue",
        "綠": "green", "綠色": "green", "green": "green",
        "白": "white", "白色": "white", "white": "white",
        "黑": "black", "黑色": "black", "black": "black",
        "黃": "yellow", "黃色": "yellow", "yellow": "yellow",
        "灰": "gray", "灰色": "gray", "gray": "gray",
        "迷彩": "camouflage", "camouflage": "camouflage",
    }

    # 目標類型關鍵字
    TYPE_KW = {
        "人員": "personnel", "人": "personnel", "男: ": "personnel",
        "女": "personnel", "person": "personnel", "人物": "personnel",
        "車輛": "vehicle", "車": "vehicle", "卡車": "vehicle",
        "轎車": "vehicle", "機車": "vehicle", "vehicle": "vehicle",
        "truck": "vehicle", "car": "vehicle",
        "通訊": "comms_tower", "天線": "comms_tower", "基站": "comms_tower",
        "設施": "facility", "建築": "building",
        "障礙": "obstacle", "障礙物": "obstacle",
    }

    # 行為標記
    BEHAVIOR_KW = [
        "可疑", "suspicious", "armed", "armed", "逃跑", "fleeing",
        "持有", "carrying", "包裹", "package", "武裝", "帶", "帶著",
    ]

    @classmethod
    def extract(cls, instruction: str) -> dict:
        """
        從指令提取目標特徵。回傳特徵字典。
        """
        lowered = instruction.lower()
        features: dict = {
            "colors":     [],
            "target_type": "unknown",
            "behaviors":  [],
            "size":       "medium",
            "confidence_threshold": 0.7,
            "raw_instruction": instruction,
        }

        # 提取顏色
        for kw, color in cls.COLOR_KW.items():
            if kw in lowered or kw in instruction:
                if color not in features["colors"]:
                    features["colors"].append(color)

        # 提取類型
        for kw, t in cls.TYPE_KW.items():
            if kw in lowered or kw in instruction:
                features["target_type"] = t
                break

        # 提取行為
        for kw in cls.BEHAVIOR_KW:
            if kw in lowered or kw in instruction:
                if kw not in features["behaviors"]:
                    features["behaviors"].append(kw)

        # 動力學與行為樹預設配置 (模擬 LLM 推論)
        if features["target_type"] == "vehicle":
            features["kinematics"] = {"speed_range_kmh": [10, 80], "typical_movement": "Roadways"}
            features["behavior_tree"] = {"on_detect": "APPROACH_50M", "on_verify": "ORBIT_TRACK", "on_lost": "EXPAND_SEARCH_100M"}
        elif features["target_type"] == "personnel":
            features["kinematics"] = {"speed_range_kmh": [0, 15], "typical_movement": "Irregular/Pedestrian"}
            features["behavior_tree"] = {"on_detect": "APPROACH_15M", "on_verify": "HOVER_AND_REPORT", "on_lost": "EXPAND_SEARCH_30M"}
        elif features["target_type"] == "comms_tower":
            features["kinematics"] = {"speed_range_kmh": [0, 0], "typical_movement": "Static"}
            features["behavior_tree"] = {"on_detect": "APPROACH_30M", "on_verify": "ORBIT_AND_SCAN", "on_lost": "REPORT_ERROR"}
        else:
            features["kinematics"] = {"speed_range_kmh": [0, 20], "typical_movement": "Unknown"}
            features["behavior_tree"] = {"on_detect": "HOVER", "on_verify": "HOVER_AND_REPORT", "on_lost": "RESUME_PATROL"}

        # 推斷 YOLO 信心度閾值（可疑目標要求更高）
        if any(b in instruction for b in ["可疑", "suspicious", "武裝", "armed"]):
            features["confidence_threshold"] = 0.82
        elif features["target_type"] == "personnel":
            features["confidence_threshold"] = 0.75

        return features


# ══════════════════════════════════════════════════════════════════════
# Target Schema 生成器
# ══════════════════════════════════════════════════════════════════════

class TargetSchemaGenerator:
    """
    動態目標特徵生成器。
    論文 4.1 節：User Input → LLM 解析 → Target Schema → Target DB
    """

    SCHEMA_TEMPLATE = {
        "id":                  "",       # 自動生成
        "database":            "targets",
        "target_id":           "",       # 自動生成
        "target_type":         "",       # 從指令提取
        "danger_level":        "Medium", # 預設中危
        "colors":              [],       # 從指令提取
        "behaviors":           [],       # 從指令提取
        "kinematics":          {},       # 動力學特徵
        "behavior_tree":       {},       # 狀態機行為樹
        "confidence_threshold":0.75,    # YOLO 閾值
        "recommended_action":  "FOLLOW_BEHAVIOR_TREE", # Phase 5 改以行為樹為主
        "chunk":               "",       # RAG 語意分塊文字
        "source":              "dynamic_generated",
        "generated_from":      "",       # 原始指令
        "timestamp":           "",
    }

    TYPE_ACTION_MAP = {
        "personnel":   "HOVER_AND_REPORT",
        "vehicle":     "FOLLOW_AND_REPORT",
        "comms_tower": "ORBIT_AND_SCAN",
        "obstacle":    "AVOID_AND_REPORT",
        "unknown":     "HOVER_AND_REPORT",
    }

    TYPE_DANGER_MAP = {
        "personnel":   "Medium",
        "vehicle":     "Medium",
        "comms_tower": "High",
        "obstacle":    "Low",
        "unknown":     "High",
    }

    TYPE_ZH = {
        "personnel":   "人員",
        "vehicle":     "車輛",
        "comms_tower": "通訊設施",
        "obstacle":    "障礙物",
        "building":    "建築物",
        "unknown":     "未知目標",
    }

    def __init__(self, targets_db_path: str = "db/targets_db.json"):
        self.db_path = Path(targets_db_path)

    def generate_from_instruction(self, instruction: str) -> dict:
        """
        論文核心：從自然語言指令動態生成 Target Schema 條目。

        Args:
            instruction: 自然語言任務指令

        Returns:
            符合 targets_db.json Schema 的條目字典
        """
        features = FeatureExtractor.extract(instruction)
        target_type = features["target_type"]
        colors      = features["colors"]
        behaviors   = features["behaviors"]
        kinematics  = features["kinematics"]
        beh_tree    = features["behavior_tree"]
        conf_thresh = features["confidence_threshold"]

        # 自動生成 ID
        _id = f"TGT-DYN-{target_type[:3].upper()}-{uuid.uuid4().hex[:4].upper()}"

        # 建立動態語意分塊（RAG 檢索用）
        color_str = "、".join(colors) if colors else "未指定"
        beh_str   = "，".join(behaviors) if behaviors else "無特殊"
        type_zh   = self.TYPE_ZH.get(target_type, target_type)
        action    = self.TYPE_ACTION_MAP.get(target_type, "HOVER_AND_REPORT")
        danger    = self.TYPE_DANGER_MAP.get(target_type, "Medium")

        # 可疑行為提升危險等級
        if any(b in behaviors for b in ["可疑", "suspicious", "武裝", "armed"]):
            danger = "High"
            action = "HOVER_AND_REPORT"

        chunk = (
            f"動態目標條目 [{_id}]：偵蒐目標類型為{type_zh}，"
            f"特徵顏色：{color_str}，行為標記：{beh_str}。"
            f"動力學：速域 {kinematics.get('speed_range_kmh')} km/h，動態 {kinematics.get('typical_movement')}。"
            f"行為樹：{json.dumps(beh_tree, ensure_ascii=False)}。"
            f"危險等級：{danger}，YOLO 信心度閾值 {conf_thresh:.0%}。"
            f"建議動作：{action}。"
            f"原始任務指令：「{instruction[:80]}」"
        )

        entry = dict(self.SCHEMA_TEMPLATE)
        entry.update({
            "id":                  _id,
            "target_id":           _id,
            "target_type":         target_type,
            "danger_level":        danger,
            "colors":              colors,
            "behaviors":           behaviors,
            "kinematics":          kinematics,
            "behavior_tree":       beh_tree,
            "confidence_threshold":conf_thresh,
            "recommended_action":  action,
            "chunk":               chunk,
            "generated_from":      instruction,
            "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        return entry

    def upsert_to_db(self, entry: dict) -> bool:
        """
        將動態生成的目標條目寫入 targets_db.json（Upsert 模式）。
        同類型的舊動態條目會被替換，靜態條目保留。

        Returns:
            True = 寫入成功
        """
        try:
            if self.db_path.exists():
                existing = json.loads(self.db_path.read_text(encoding="utf-8"))
            else:
                existing = []

            # 移除同 target_type 的舊動態條目（避免重複）
            new_type = entry.get("target_type")
            filtered = [
                e for e in existing
                if not (e.get("source") == "dynamic_generated" and
                        e.get("target_type") == new_type)
            ]
            filtered.append(entry)

            self.db_path.write_text(
                json.dumps(filtered, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"  [TargetSchema] 已寫入動態條目 {entry['id']} → {self.db_path}")
            return True

        except Exception as e:
            print(f"  [TargetSchema] 寫入失敗: {e}")
            return False


if __name__ == "__main__":
    gen = TargetSchemaGenerator(targets_db_path="db/targets_db.json")

    for instruction in [
        "搜索綠衣可疑人員，可能攜帶武器",
        "追蹤紅色可疑車輛，疑似未知逃亡目標",
        "偵察通訊天線基站，確認是否受損",
    ]:
        print(f"\n指令: 「{instruction}」")
        entry = gen.generate_from_instruction(instruction)
        print(f"  類型: {entry['target_type']} | 顏色: {entry['colors']} | 危險: {entry['danger_level']}")
        print(f"  分塊: {entry['chunk'][:80]}...")
        gen.upsert_to_db(entry)
