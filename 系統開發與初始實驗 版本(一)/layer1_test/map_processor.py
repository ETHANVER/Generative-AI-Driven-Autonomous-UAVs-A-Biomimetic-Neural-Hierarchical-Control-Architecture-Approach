"""
map_processor.py - AirSim RT-EXR → 語意網格地圖 → environment_db.json
=========================================================================
根據論文第 4.1 節「環境資料庫建置流程」：
  AirSim RT (Render Target) EXR 圖資
    → 網格切割 (Grid Partitioning)
    → 亮度/反射分析 (Luminance Analysis, 模擬 CNN 分類)
    → 語意地圖 (Global Semantic Map, JSON)
    → 填入 environment_db.json

★ 針對 AirSim RT EXR 的設計要點：
  - AirSim RT EXR 是 HDR「反射/場景捕捉」圖，非一般 RGB 照片
  - RGB 通道值域約 0~0.74（極暗 HDR float，需 Log 色調映射）
  - 零值區 = 天空/空曠 → Open_Field
  - 高亮度區 = 建築頂面 → Building_Zone
  - 中等亮度紅色偏移 = 植被 → Forest/Vegetation
  - 中等均勻亮度 = 道路/地面 → Road
  - 使用 --merge 參數保留原有手動分塊並加入自動分塊

使用方式：
  python map_processor.py                     # 處理 Map_RT.EXR
  python map_processor.py --grid 10 --vis     # 10×10 格，輸出視覺化
  python map_processor.py --merge             # 保留舊手動條目 + 加入新自動條目
  python map_processor.py --input other.png   # 也支援 PNG/JPG
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# EXR 讀取（多後端，針對 AirSim RT 格式）
# ══════════════════════════════════════════════════════════════════════

def read_image(path: str) -> np.ndarray:
    """
    讀取圖像並回傳 float32 (H,W,3)，值域 [0,1]。
    自動偵測 EXR vs 一般圖像。
    """
    path = str(path)
    ext  = Path(path).suffix.upper()

    if ext == ".EXR":
        return _read_exr(path)
    else:
        return _read_normal(path)


def _read_exr(path: str) -> np.ndarray:
    """讀取 AirSim RT EXR，套用 Log 色調映射"""
    import warnings; warnings.filterwarnings("ignore")
    errors = []

    # ── 後端 1：imageio FreeImage ──────────────────────────
    try:
        import imageio
        try: imageio.plugins.freeimage.download()
        except: pass
        img = np.array(imageio.imread(path, format='EXR-FI'), dtype=np.float32)
        print(f"  [EXR reader] imageio FreeImage — shape={img.shape}")
        return _tone_map_airsim(img)
    except Exception as e:
        errors.append(f"imageio FreeImage: {e}")

    # ── 後端 2：imageio v3 ─────────────────────────────────
    try:
        import imageio.v3 as iio
        img = np.array(iio.imread(path), dtype=np.float32)
        if img.size > 0:
            print(f"  [EXR reader] imageio v3 — shape={img.shape}")
            return _tone_map_airsim(img)
    except Exception as e:
        errors.append(f"imageio v3: {e}")

    # ── 後端 3：OpenEXR ─────────────────────────────────────
    try:
        import OpenEXR, Imath
        f = OpenEXR.InputFile(path)
        hdr = f.header()
        dw  = hdr['dataWindow']
        W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        FT  = Imath.PixelType(Imath.PixelType.FLOAT)
        chs = list(hdr['channels'].keys())
        def get_ch(names):
            for n in names:
                for c in chs:
                    if c.upper() == n.upper():
                        return c
            return chs[0]
        r = np.frombuffer(f.channel(get_ch(['R','Y']), FT), np.float32).reshape(H,W)
        g = np.frombuffer(f.channel(get_ch(['G','RY']),FT), np.float32).reshape(H,W)
        b = np.frombuffer(f.channel(get_ch(['B','BY']),FT), np.float32).reshape(H,W)
        img = np.stack([r,g,b], axis=2)
        print(f"  [EXR reader] OpenEXR — shape={img.shape}, channels={chs}")
        return _tone_map_airsim(img)
    except Exception as e:
        errors.append(f"OpenEXR: {e}")

    print("⚠️  所有 EXR 後端失敗，使用模擬圖像")
    for e in errors: print(f"   {e}")
    return _synthetic_map()


def _tone_map_airsim(img: np.ndarray) -> np.ndarray:
    """
    針對 AirSim RT EXR 的色調映射。
    AirSim RT 圖像特性：
      - 大量零值像素（天空/空曠）
      - 有效值集中在 0~0.74 的極暗 HDR 浮點數
      - 使用 log(1 + k*img) 展開暗部細節
    """
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    rgb = img[:, :, :3]

    # 計算有效像素範圍
    max_val = rgb.max()
    mean_nonzero = rgb[rgb > 0].mean() if (rgb > 0).any() else 0.01

    # 自適應增益係數（讓 p95 ≈ 0.8）
    p95   = float(np.percentile(rgb[rgb > 0], 95)) if (rgb > 0).any() else 0.01
    gain  = 0.8 / (p95 + 1e-8)
    gain  = min(gain, 500.0)   # 上限避免爆炸

    # Log 色調映射：log(1 + gain * rgb) / log(1 + gain)  
    rgb_tm = np.log1p(gain * rgb) / np.log1p(gain)
    rgb_tm = np.clip(rgb_tm, 0.0, 1.0)

    print(f"  [ToneMap] gain={gain:.1f}, p95={p95:.5f},  "
          f"mapped_max={rgb_tm.max():.3f}, mapped_mean={rgb_tm.mean():.4f}")
    return rgb_tm.astype(np.float32)


def _read_normal(path: str) -> np.ndarray:
    """讀取普通 PNG/JPG"""
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return (img / 255.0).astype(np.float32)
    except: pass
    try:
        from PIL import Image
        import numpy as np
        img = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
        return img / 255.0
    except: pass
    return _synthetic_map()


def _synthetic_map(size: int = 512) -> np.ndarray:
    """模擬 AirSim 俯拍測試圖像（EXR 讀取失敗備用）"""
    rng = np.random.default_rng(42)
    h = w = size
    img = np.ones((h, w, 3), dtype=np.float32) * 0.18  # 預設：街道灰

    # 森林（左上）
    img[:h//2, :w//2]  = [0.08, 0.28, 0.06]
    img[:h//2, :w//2] += rng.normal(0, 0.02, (h//2, w//2, 3))

    # 建築（右上，高亮）
    img[:h//2, w//2:]  = [0.55, 0.55, 0.58]
    img[:h//2, w//2:] += rng.normal(0, 0.03, (h//2, w//2, 3))

    # 道路（中央十字）
    img[h//2-3:h//2+3, :]  = [0.30, 0.30, 0.32]
    img[:, w//2-3:w//2+3]  = [0.30, 0.30, 0.32]

    # 水域（右下）
    img[h//2:, w//2:]  = [0.10, 0.18, 0.52]

    print(f"  [Synthetic] 生成 {size}×{size} 合成測試圖")
    return np.clip(img, 0, 1)


# ══════════════════════════════════════════════════════════════════════
# 網格切割 + 特徵提取
# ══════════════════════════════════════════════════════════════════════

MAP_COVERAGE_M = 200.0   # 圖像覆蓋的實際距離（公尺）

def partition_grid(img: np.ndarray, grid_n: int) -> list:
    h, w = img.shape[:2]
    cell_h, cell_w = h // grid_n, w // grid_n
    cells = []
    for row in range(grid_n):
        for col in range(grid_n):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            patch = img[y1:y2, x1:x2]

            r, g, b = patch[:,:,0], patch[:,:,1], patch[:,:,2]
            lum   = 0.2126*r + 0.7152*g + 0.0722*b   # 亮度 (ITU-R BT.709)
            saturation = (patch.max(axis=2) - patch.min(axis=2)) / \
                         (patch.max(axis=2) + 1e-6)

            cells.append({
                "row": row, "col": col,
                "cx_m": round((col + 0.5) / grid_n * MAP_COVERAGE_M, 1),
                "cy_m": round((row + 0.5) / grid_n * MAP_COVERAGE_M, 1),
                "mean_r":  float(r.mean()),
                "mean_g":  float(g.mean()),
                "mean_b":  float(b.mean()),
                "lum":     float(lum.mean()),
                "lum_std": float(lum.std()),
                "sat":     float(saturation.mean()),
                # 衍生特徵
                "greenness": float(g.mean() - (r.mean() + b.mean()) / 2),
                "blueness":  float(b.mean() - (r.mean() + g.mean()) / 2),
                "darkness":  float(1.0 - lum.mean()),
                "zero_ratio":float((lum < 0.02).mean()),  # 零/極暗像素佔比（AirSim sky）
            })
    return cells


# ══════════════════════════════════════════════════════════════════════
# 語意分類（針對 AirSim RT EXR 調校）
# ══════════════════════════════════════════════════════════════════════

def classify_cell(cell: dict) -> tuple:
    """
    回傳 (area_type, danger_level, confidence)

    AirSim RT EXR 分類策略（Log 色調映射後）：
      - zero_ratio > 0.9  → Sky/Open_Field（幾乎全黑 = 天空佔位格）
      - lum > 0.4         → Building_Zone（高反射屋頂）
      - greenness > 0.04  → Forest
      - blueness > 0.04   → Water
      - lum_std > 0.12    → Building_Zone（高對比 = 建築邊緣）
      - lum 0.1~0.3       → Road/Ground
      - else              → Open_Field
    """
    lum      = cell["lum"]
    lum_std  = cell["lum_std"]
    green    = cell["greenness"]
    blue     = cell["blueness"]
    zero_r   = cell["zero_ratio"]
    sat      = cell["sat"]

    # ── 天空/空曠格子（AirSim EXR 特有：大量零像素 = 俯拍空區）
    if zero_r > 0.85:
        return "Open_Field", "Low", 0.70

    # ── 高亮 → 建築屋頂（高反射）
    if lum > 0.38:
        return "Building_Zone", "High", min(0.95, 0.70 + lum)

    # ── 高飽和綠 → 植被
    if green > 0.05 and sat > 0.10:
        return "Forest", "Medium", min(0.92, 0.65 + green * 5)
    if green > 0.03:
        return "Vegetation", "Low", min(0.85, 0.60 + green * 5)

    # ── 高飽和藍 → 水域
    if blue > 0.05 and sat > 0.08:
        return "Water", "Low", min(0.90, 0.65 + blue * 5)

    # ── 高對比（邊緣信息）→ 建築物
    if lum_std > 0.10:
        return "Building_Zone", "High", 0.72

    # ── 中等亮度均勻 → 道路/地面
    if 0.10 < lum <= 0.30 and lum_std < 0.08:
        return "Road", "Low", 0.68

    # ── 中低亮度 → 開闊地
    if lum <= 0.20:
        return "Open_Field", "Low", 0.65

    return "Unknown", "Medium", 0.50


# ══════════════════════════════════════════════════════════════════════
# 高度配置 + 說明
# ══════════════════════════════════════════════════════════════════════

ALT_CONFIG = {
    "Forest":        {"min": -20.0, "max": -10.0, "rec": -15.0},
    "Vegetation":    {"min": -15.0, "max": -8.0,  "rec": -12.0},
    "Building_Zone": {"min": -60.0, "max": -50.0, "rec": -55.0},
    "Road":          {"min": -12.0, "max": -5.0,  "rec": -8.0},
    "Water":         {"min": -20.0, "max": -5.0,  "rec": -10.0},
    "Open_Field":    {"min": -30.0, "max": -5.0,  "rec": -10.0},
    "Unknown":       {"min": -25.0, "max": -15.0, "rec": -20.0},
}
TYPE_ZH = {
    "Forest":        "森林植被區（中度危險，通訊干擾風險）",
    "Vegetation":    "植被覆蓋區（低危險，視線受阻）",
    "Building_Zone": "建築密集區（高危險，禁航區）",
    "Road":          "道路/地面（低危險，可飛行走廊）",
    "Water":         "水域（低危險，無地面障礙）",
    "Open_Field":    "開闊平原（低危險，高速巡航理想）",
    "Unknown":       "未分類區域（需人工確認）",
}


# ══════════════════════════════════════════════════════════════════════
# 語意地圖生成
# ══════════════════════════════════════════════════════════════════════

def build_semantic_map(cells: list) -> list:
    result = []
    for cell in cells:
        t, danger, conf = classify_cell(cell)
        alt = ALT_CONFIG.get(t, ALT_CONFIG["Open_Field"])
        result.append({
            "grid_row": cell["row"],
            "grid_col": cell["col"],
            "cx_m": cell["cx_m"],
            "cy_m": cell["cy_m"],
            "area_type": t,
            "danger_level": danger,
            "confidence": round(conf, 3),
            "lum": round(cell["lum"], 4),
            "greenness": round(cell["greenness"], 4),
            "zero_ratio": round(cell["zero_ratio"], 4),
            "alt_min": alt["min"],
            "alt_max": alt["max"],
            "alt_recommended": alt["rec"],
            "restriction": "No-Fly" if t == "Building_Zone" else None,
        })
    return result


# ══════════════════════════════════════════════════════════════════════
# environment_db.json 分塊生成
# ══════════════════════════════════════════════════════════════════════

def generate_env_chunks(semantic_map: list, grid_n: int) -> list:
    """將語意地圖轉換為 RAG 語意分塊格式（Semantic Chunking）"""
    # 按類型分組
    groups: dict = {}
    for e in semantic_map:
        t = e["area_type"]
        groups.setdefault(t, []).append(e)

    chunks = []
    HALF = MAP_COVERAGE_M / grid_n / 2

    for area_type, entries in groups.items():
        area_id = f"RT_{area_type[:3].upper()}_01"
        x_vals = [e["cx_m"] for e in entries]
        y_vals = [e["cy_m"] for e in entries]
        boundary = {
            "west":  round(min(x_vals) - HALF, 1),
            "east":  round(max(x_vals) + HALF, 1),
            "south": round(min(y_vals) - HALF, 1),
            "north": round(max(y_vals) + HALF, 1),
        }
        avg_conf  = round(sum(e["confidence"] for e in entries) / len(entries), 3)
        alt = ALT_CONFIG.get(area_type, ALT_CONFIG["Open_Field"])
        danger = entries[0]["danger_level"]
        restriction = "No-Fly" if area_type == "Building_Zone" else None

        # 分塊 1：類型描述
        chunks.append({
            "id":          f"ENV-{area_id}-TYPE",
            "database":    "environment",
            "area_id":     area_id,
            "area_type":   area_type,
            "danger_level":danger,
            "confidence":  avg_conf,
            "grid_count":  len(entries),
            "chunk": (
                f"{area_id} 是{TYPE_ZH.get(area_type, area_type)}，"
                f"屬於{danger}危險等級，共佔 {len(entries)} 個網格格子，"
                f"由 AirSim EXR 圖資自動分類，信心度 {avg_conf:.0%}。"
            ),
            "source": "auto_generated_from_airsim_exr",
        })

        # 分塊 2：座標邊界
        cx = round((boundary['west']+boundary['east'])/2, 1)
        cy = round((boundary['south']+boundary['north'])/2, 1)
        chunks.append({
            "id":       f"ENV-{area_id}-BOUNDARY",
            "database": "environment",
            "area_id":  area_id,
            "chunk": (
                f"{area_id} 邊界（AirSim NED）："
                f"西 {boundary['west']}m，東 {boundary['east']}m，"
                f"南 {boundary['south']}m，北 {boundary['north']}m。"
                f"座標中心約 ({cx}, {cy})。"
            ),
            "boundary": boundary,
            "source": "auto_generated_from_airsim_exr",
        })

        # 分塊 3：高度建議
        chunks.append({
            "id":       f"ENV-{area_id}-ALT",
            "database": "environment",
            "area_id":  area_id,
            "chunk": (
                f"{area_id} ({area_type}) 建議飛行高度："
                f"最低 {abs(alt['max'])}m，最高 {abs(alt['min'])}m"
                f"（NED {alt['max']} 至 {alt['min']}），"
                f"最佳高度 {abs(alt['rec'])}m。"
            ),
            "recommended_altitude_min": alt["min"],
            "recommended_altitude_max": alt["max"],
            "source": "auto_generated_from_airsim_exr",
        })

        # 分塊 4（高危禁航）
        if restriction == "No-Fly":
            chunks.append({
                "id":         f"ENV-{area_id}-RESTRICTION",
                "database":   "environment",
                "area_id":    area_id,
                "restriction":"No-Fly",
                "chunk": (
                    f"{area_id} 為禁航區 (No-Fly Zone)：建築密集，"
                    f"Layer 1 飛入該區的決策必須被 Layer 2 Geofencing 攔截，"
                    f"並寫入情節記憶為負面事件。"
                ),
                "source": "auto_generated_from_airsim_exr",
            })

    # 全域摘要分塊
    type_summary = ", ".join(f"{t}({len(v)}格)" for t, v in groups.items())
    chunks.append({
        "id":         "ENV-GLOBAL-AIRSIM-RT-MAP",
        "database":   "environment",
        "area_id":    "GLOBAL",
        "chunk": (
            f"AirSim RT-EXR 語意網格地圖（{grid_n}×{grid_n} 格，共 {len(semantic_map)} 格）："
            f"由高空俯拍渲染目標圖資自動分類生成。"
            f"地形分佈：{type_summary}。"
            f"覆蓋範圍：{MAP_COVERAGE_M}m×{MAP_COVERAGE_M}m。AirSim NED 座標系。"
        ),
        "grid_size":   grid_n,
        "total_cells": len(semantic_map),
        "type_counts": {t: len(v) for t, v in groups.items()},
        "source": "auto_generated_from_airsim_exr",
    })

    return chunks


# ══════════════════════════════════════════════════════════════════════
# 視覺化
# ══════════════════════════════════════════════════════════════════════

PALETTE = {
    "Forest":        (0.13, 0.55, 0.13),
    "Vegetation":    (0.56, 0.93, 0.56),
    "Building_Zone": (0.50, 0.50, 0.52),
    "Road":          (0.80, 0.80, 0.80),
    "Water":         (0.20, 0.45, 0.85),
    "Open_Field":    (0.85, 0.82, 0.55),
    "Unknown":       (0.65, 0.10, 0.65),
}


def visualize(sem_map: list, grid_n: int, out: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
        fig.patch.set_facecolor('#1a1a2e')
        fig.suptitle(
            "AirSim RT-EXR 語意網格地圖\nSemantic Classification from Aerial RT Image",
            fontsize=13, color='white', fontweight='bold'
        )

        for ax in axes:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')

        # 左：語意地圖
        ax = axes[0]
        grid_img = np.zeros((grid_n, grid_n, 3))
        conf_grid = np.zeros((grid_n, grid_n))
        for e in sem_map:
            r, c = e["grid_row"], e["grid_col"]
            grid_img[r, c] = PALETTE.get(e["area_type"], (0.5,0.5,0.5))
            conf_grid[r, c] = e["confidence"]

        ax.imshow(grid_img, origin='upper', interpolation='nearest')
        ax.set_title(f"語意分類結果 ({grid_n}×{grid_n})", color='white', fontsize=11)
        ax.set_xlabel("X → (東)", color='gray', fontsize=9)
        ax.set_ylabel("↑ Y (北)", color='gray', fontsize=9)

        for i in range(grid_n + 1):
            ax.axhline(i - 0.5, color='white', lw=0.3, alpha=0.4)
            ax.axvline(i - 0.5, color='white', lw=0.3, alpha=0.4)

        for e in sem_map:
            r, c = e["grid_row"], e["grid_col"]
            ax.text(c, r, f"{e['area_type'][:3]}\n{e['confidence']:.0%}",
                    ha='center', va='center', fontsize=4.5,
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.45))

        patches = [mpatches.Patch(color=PALETTE[t], label=t) for t in PALETTE]
        ax.legend(handles=patches, loc='lower right', fontsize=7,
                  framealpha=0.7, labelcolor='white',
                  facecolor='#1a1a2e', edgecolor='#555')

        # 右：信心度熱圖
        ax2 = axes[1]
        im = ax2.imshow(conf_grid, origin='upper', cmap='YlOrRd',
                        vmin=0.5, vmax=1.0, interpolation='nearest')
        cb = plt.colorbar(im, ax=ax2)
        cb.ax.yaxis.set_tick_params(color='white')
        cb.set_label('信心度 (Confidence)', color='white')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
        ax2.set_title("分類信心度熱力圖", color='white', fontsize=11)
        ax2.set_xlabel("X → (東)", color='gray', fontsize=9)
        ax2.set_ylabel("↑ Y (北)", color='gray', fontsize=9)
        for i in range(grid_n + 1):
            ax2.axhline(i - 0.5, color='white', lw=0.3, alpha=0.4)
            ax2.axvline(i - 0.5, color='white', lw=0.3, alpha=0.4)
        for e in sem_map:
            r, c = e["grid_row"], e["grid_col"]
            ax2.text(c, r, f"{e['confidence']:.0%}",
                     ha='center', va='center', fontsize=6, color='black', fontweight='bold')

        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        print(f"  ✅ 視覺化: {out}")
    except Exception as e:
        print(f"  ⚠️  視覺化失敗: {e}")


# ══════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="AirSim EXR → environment_db.json")
    ap.add_argument("--input",      default="Map_RT.EXR")
    ap.add_argument("--grid",       type=int, default=8)
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--vis",        action="store_true")
    ap.add_argument("--merge",      action="store_true",
                    help="保留 db/environment_db.json 中的手動條目")
    args = ap.parse_args()

    print("=" * 62)
    print("  AirSim RT-EXR 圖資處理器 v2")
    print("  論文：基於仿生神經分工架構之生成式 AI 自主無人機研究")
    print("=" * 62)

    # Step 1: 讀取
    print(f"\n[Step 1] 讀取: {args.input}")
    t0  = time.time()
    img = read_image(args.input)
    print(f"  → shape={img.shape}, 耗時 {time.time()-t0:.2f}s")
    print(f"  → 值域: [{img.min():.4f}, {img.max():.4f}], mean={img.mean():.4f}")

    # Step 2: 網格切割
    print(f"\n[Step 2] 網格切割 ({args.grid}×{args.grid})...")
    cells = partition_grid(img, args.grid)

    # Step 3: 分類
    print(f"\n[Step 3] 語意分類...")
    sem_map = build_semantic_map(cells)
    counts: dict = {}
    for e in sem_map:
        counts[e["area_type"]] = counts.get(e["area_type"], 0) + 1
    for t, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * n + "░" * (args.grid**2 // 8 - n//2)
        print(f"  {t:20s} [{e['danger_level']:6s}] {bar[:20]} ({n}格, "
              f"{n/len(sem_map):.0%})")

    # Step 4: 生成分塊
    print(f"\n[Step 4] 生成 RAG 語意分塊...")
    chunks = generate_env_chunks(sem_map, args.grid)
    print(f"  → {len(chunks)} 個語意分塊")

    # 儲存
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

    sm_path = out / "semantic_map.json"
    sm_path.write_text(json.dumps(sem_map, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n  💾 語意地圖 → {sm_path}")

    db_dir  = out / "db"
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "environment_db.json"

    if args.merge and db_path.exists():
        existing = json.loads(db_path.read_text(encoding='utf-8'))
        manual   = [e for e in existing if e.get("source") != "auto_generated_from_airsim_exr"]
        final_db = manual + chunks
        print(f"  🔀 合併: {len(manual)} 手動 + {len(chunks)} 自動 = {len(final_db)} 分塊")
    else:
        final_db = chunks

    db_path.write_text(json.dumps(final_db, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  💾 environment_db.json → {db_path} ({len(final_db)} 分塊)")

    if args.vis:
        print(f"\n[Step 5] 視覺化...")
        visualize(sem_map, args.grid, str(out / "semantic_map_vis.png"))

    print(f"\n{'='*62}")
    print(f"  ✅ 完成！")
    print(f"  → 下一步：python layer1_standalone.py 驗證 RAG 查詢效果")
    print(f"  → 使用 --merge 可保留並合併手動設定的條目")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
