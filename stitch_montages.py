#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper layout A montage builder.

Layout per Country x Step:
  Top:    1x4  (real-only models)    Naive | ARIMA | TabPFN_ts | Respicast
  Bottom: 3x3  (tri-data models)     rows=real/augmented/combined, cols=DLinear/LSTM/Autoformer

Reads single-step images from:
  epi4cast/test_results/<folder>/rolling_test_step{k}.png   (NO underscore)

Folder naming expected (examples):
  DLinear_real_Ireland_ILI
  LSTM_augmented_Italy_ILI
  Autoformer_combined_France_ILI
  Naive_real_Belgium_ILI
  ARIMA_real_Belgium_ILI
  TabPFN_ts_real_Belgium_ILI
  Respicast_real_Belgium_ILI

Output:
  epi4cast/test_results/montages/<Country>/rolling_test_step{k}.png
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# -------- config --------
TRI_MODELS = ["DLinear", "LSTM", "Autoformer"]
REAL_ONLY_MODELS = ["ARIMA", "TabPFN_ts", "Respicast"] # "Naive",
SPLITS = ["real", "augmented", "combined"]
STEPS = [1, 2, 3, 4]
SIM_GRID = [
    ["Naive", "ARIMA"],
    ["DLinear", "LSTM"],
    ["Autoformer", "TabPFN_ts"],
]
def fname_step(k: int) -> str:
    return f"rolling_test_step{k}.png"  # ✅ no underscore

# ---------- utilities ----------
def safe_name(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", str(s))
    return s.strip().strip(".") or "unknown"

def get_font(size: int = 22, bold: bool = False):
    # try common fonts on Windows; fallback to default
    try:
        if bold:
            return ImageFont.truetype("arialbd.ttf", size=size)
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def load_img(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")

def make_placeholder(size: Tuple[int, int], text: str) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    f = get_font(22, bold=True)
    d.rectangle([0, 0, w-1, h-1], outline=(200, 200, 200), width=2)
    d.text((16, 14), text, fill=(60, 60, 60), font=f)
    return img

def overlay_title(img: Image.Image, title: str) -> Image.Image:
    """Left-top bold title on each tile, avoid y-axis area."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    f = get_font(100, bold=True)  # 100 没问题

    # 给 y-axis 留空间：把标题整体右移、下移
    x, y = 80, 18   # 你可以从 x=70~110 之间试，y=12~25 之间试

    # 可选：白底，提高可读性（避免压到曲线/点）
    bbox = d.textbbox((x, y), title, font=f)
    pad = 6
    d.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad], fill=(255, 255, 255))

    # 伪加粗（你原来的两次绘制）
    d.text((x, y), title, fill=(0, 0, 0), font=f)
    d.text((x+1, y), title, fill=(0, 0, 0), font=f)
    return img

def add_header(canvas: Image.Image, text: str, height: int = 46) -> Image.Image:
    w, h = canvas.size
    out = Image.new("RGB", (w, h + height), (255, 255, 255))
    out.paste(canvas, (0, height))
    d = ImageDraw.Draw(out)
    f = get_font(28, bold=True)
    d.text((14, 8), text, fill=(0,0,0), font=f)
    return out

def add_footer_legend(canvas: Image.Image, height: int = 92) -> Image.Image:
    """
    Bottom legend: centered and larger.
    Items:
      Truth (black line + dot)
      Prediction (mean + 80% PI) (band + line)
      Split (gray dashed vertical)
    """
    w, h = canvas.size
    out = Image.new("RGB", (w, h + height), (255, 255, 255))
    out.paste(canvas, (0, 0))

    # bigger font
    f = get_font(72, bold=True)

    # paper colors (match your visual)
    axis_black = (20, 20, 20)
    pred = (46, 111, 78)     # #2E6F4E #(31, 78, 121)     # #1F4E79
    split = (140, 140, 140)  # #8C8C8C
    alpha = 0.18

    y = h + (height // 2) - 16  # vertically centered-ish

    # --- measure widths to center ---
    d = ImageDraw.Draw(out)

    def text_w(s: str) -> int:
        try:
            return int(d.textlength(s, font=f))
        except Exception:
            # fallback
            return int(f.getlength(s)) if hasattr(f, "getlength") else len(s) * 14

    gap = 60  # spacing between legend items

    # symbol widths
    truth_sym_w = 60
    pred_sym_w = 62
    split_sym_w = 40

    lab_truth = "Truth"
    lab_pred = "Prediction (mean + 80% PI)"
    lab_split = "Split"

    total_w = (
        truth_sym_w + 14 + text_w(lab_truth) +
        gap +
        pred_sym_w + 14 + text_w(lab_pred) +
        gap +
        split_sym_w + 14 + text_w(lab_split)
    )

    x = max(20, (w - total_w) // 2)

    # ---------- draw Truth ----------
    x0 = x
    d.line((x0, y + 18, x0 + 52, y + 18), fill=axis_black, width=5)
    d.ellipse((x0 + 22, y + 14, x0 + 30, y + 22), fill=axis_black)
    d.text((x0 + truth_sym_w + 14, y + 6), lab_truth, fill=axis_black, font=f)
    x = x0 + truth_sym_w + 14 + text_w(lab_truth) + gap

    # ---------- draw Prediction (band + line) ----------
    bx0, by0 = x, y + 10
    bw, bh = 62, 22

    band = Image.new("RGBA", (bw, bh), (pred[0], pred[1], pred[2], int(255 * alpha)))
    out_rgba = out.convert("RGBA")
    out_rgba.paste(band, (bx0, by0), band)
    d2 = ImageDraw.Draw(out_rgba)
    midy = by0 + bh // 2
    d2.line((bx0, midy, bx0 + bw, midy), fill=(pred[0], pred[1], pred[2], 255), width=5)
    out = out_rgba.convert("RGB")
    d = ImageDraw.Draw(out)

    d.text((x + pred_sym_w + 14, y + 6), lab_pred, fill=axis_black, font=f)
    x = x + pred_sym_w + 14 + text_w(lab_pred) + gap

    # ---------- draw Split (gray dashed vertical) ----------
    sx = x + 16
    for yy in range(y + 6, y + 34, 6):
        d.line((sx, yy, sx, yy + 3), fill=split, width=4)
    d.text((x + split_sym_w + 14, y + 6), lab_split, fill=axis_black, font=f)

    return out

# ---------- folder matching ----------
def find_folder(test_results_dir: Path, model: str, split: str, country: str) -> Optional[Path]:
    """
    Expect folders like:
      <Model>_<split>_<Country>_ILI
      <Model>_<split>_<Country>_ILI_median (possible)
    For this montage:
      - real-only models always use split='real'
      - tri models use split in {real,augmented,combined}
    """
    suffixes = ["_ILI", "_ILI_median"]
    cands = []
    for suf in suffixes:
        cands.append(f"{model}_{split}_{country}{suf}")

    for name in cands:
        p = test_results_dir / name
        if p.exists() and p.is_dir():
            return p

    # fuzzy fallback
    ml = model.lower()
    sl = split.lower()
    cl = country.lower()
    for p in test_results_dir.iterdir():
        if not p.is_dir():
            continue
        s = p.name.lower()
        if ml in s and sl in s and cl in s:
            return p
    return None

def detect_repo_root(repo_root_arg: str) -> Path:
    root = Path(repo_root_arg).resolve()
    if (root / "test_results").exists():
        return root
    if (root / "epi4cast" / "test_results").exists():
        return root / "epi4cast"
    # fallback: script dir
    sd = Path(__file__).resolve().parent
    if (sd / "test_results").exists():
        return sd
    if (sd / "epi4cast" / "test_results").exists():
        return sd / "epi4cast"
    raise FileNotFoundError(f"Cannot locate epi4cast root from: {root}")

def infer_countries(test_results_dir: Path) -> List[str]:
    countries = set()
    for p in test_results_dir.iterdir():
        if not p.is_dir():
            continue
        # capture ..._<Country>_ILI...
        parts = p.name.split("_")
        if "ILI" in parts:
            idx = parts.index("ILI")
            if idx - 1 >= 0:
                countries.add(parts[idx - 1])
    return sorted(countries)

# ---------- montage building ----------
def hstack(images: List[Image.Image], pad: int = 12, bg=(255,255,255)) -> Image.Image:
    h = max(im.size[1] for im in images)
    w = sum(im.size[0] for im in images) + pad * (len(images)+1)
    out = Image.new("RGB", (w, h + 2*pad), bg)
    x = pad
    y = pad
    for im in images:
        out.paste(im, (x, y))
        x += im.size[0] + pad
    return out

def vstack(images: List[Image.Image], pad: int = 12, bg=(255,255,255)) -> Image.Image:
    w = max(im.size[0] for im in images)
    h = sum(im.size[1] for im in images) + pad * (len(images)+1)
    out = Image.new("RGB", (w + 2*pad, h), bg)
    x = pad
    y = pad
    for im in images:
        out.paste(im, (x, y))
        y += im.size[1] + pad
    return out

def build_country_step_montage(test_results_dir: Path, country: str, k: int,
                              tile_size: Tuple[int,int]) -> Image.Image:
    # --- Top strip (1x4): real-only models, real split only ---
    top_tiles = []
    for model in REAL_ONLY_MODELS:
        folder = find_folder(test_results_dir, model, "real", country)
        img_path = folder / fname_step(k) if folder else None
        if img_path and img_path.exists():
            im = load_img(img_path)
            if im.size != tile_size:
                im = im.resize(tile_size)
        else:
            im = make_placeholder(tile_size, f"Missing {model}_real")
        im = overlay_title(im, f"{model}_real")
        top_tiles.append(im)
    top_row = hstack(top_tiles, pad=12)

    # --- Bottom block (3x3): tri models × splits ---
    block_rows = []
    for split in SPLITS:
        row_tiles = []
        for model in TRI_MODELS:
            folder = find_folder(test_results_dir, model, split, country)
            img_path = folder / fname_step(k) if folder else None
            if img_path and img_path.exists():
                im = load_img(img_path)
                if im.size != tile_size:
                    im = im.resize(tile_size)
            else:
                im = make_placeholder(tile_size, f"Missing {model}_{split}")
            title = f"{model}_{'aug' if split=='augmented' else ('comb' if split=='combined' else 'real')}"
            im = overlay_title(im, title)
            row_tiles.append(im)
        block_rows.append(hstack(row_tiles, pad=12))
    bottom_block = vstack(block_rows, pad=12)

    # Stack: top then bottom
    montage = vstack([top_row, bottom_block], pad=14)
    return montage
def build_country_step_montage_simulated_3x2(test_results_dir: Path, country: str, k: int,
                                             tile_size: Tuple[int,int]) -> Image.Image:
    """3x2 montage for simulated split:
       Row1: Naive_simulated | ARIMA_simulated
       Row2: DLinear_simulated | LSTM_simulated
       Row3: Autoformer_simulated | TabPFN_ts_simulated
    """
    block_rows = []
    for row in SIM_GRID:
        row_tiles = []
        for model in row:
            folder = find_folder(test_results_dir, model, "simulated", country)
            img_path = folder / fname_step(k) if folder else None

            if img_path and img_path.exists():
                im = load_img(img_path)
                if im.size != tile_size:
                    im = im.resize(tile_size)
            else:
                im = make_placeholder(tile_size, f"Missing {model}_simulated")

            im = overlay_title(im, f"{model}_simulated")
            row_tiles.append(im)

        block_rows.append(hstack(row_tiles, pad=12))

    montage = vstack(block_rows, pad=14)
    return montage
# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".", help="Path to epi4cast folder or its parent.")
    ap.add_argument("--countries", nargs="*", default=None,
                    help="Countries to process (e.g., Belgium Czech Denmark ...)")
    ap.add_argument("--steps", nargs="*", type=int, default=STEPS)
    ap.add_argument("--only_simulated", action="store_true",
                    help="Only build simulated 3x2 montage (skip paper layout A).")
    args = ap.parse_args()

    repo = detect_repo_root(args.repo_root)
    test_results_dir = repo / "test_results"
    out_base = test_results_dir / "montages"
    out_base.mkdir(parents=True, exist_ok=True)

    countries = args.countries or infer_countries(test_results_dir)
    if not countries:
        raise FileNotFoundError(f"Could not infer countries from {test_results_dir}. Use --countries ...")

    # determine tile size from first existing image
    tile_size = None
    for country in countries:
        for k in args.steps:
            # try any likely folder
            for model in (REAL_ONLY_MODELS + TRI_MODELS):
                folder = find_folder(test_results_dir, model, "real", country) or \
                         find_folder(test_results_dir, model, "augmented", country) or \
                         find_folder(test_results_dir, model, "combined", country)
                if folder:
                    p = folder / fname_step(k)
                    if p.exists():
                        tile_size = load_img(p).size
                        break
            if tile_size:
                break
        if tile_size:
            break
    if tile_size is None:
        tile_size = (900, 540)  # fallback

    for country in countries:
        out_country = out_base / safe_name(country)
        out_country.mkdir(parents=True, exist_ok=True)

        for k in args.steps:
            if args.only_simulated:
                # 只做 simulated 3x2（你前面新加的函数）
                montage = build_country_step_montage_simulated_3x2(test_results_dir, country, k, tile_size)
                montage = add_footer_legend(montage)

                # 保存到 montages 里，文件名保持 rolling_test_step{k}.png
                out_sim = out_country / "simulated"
                out_sim.mkdir(parents=True, exist_ok=True)
                out_path = out_sim / fname_step(k)
                montage.save(out_path, format="PNG")
                print(f"[OK] {out_path}")
                continue

            # --- 原来的 paper layout A 不变 ---
            montage = build_country_step_montage(test_results_dir, country, k, tile_size)
            montage = add_footer_legend(montage)
            out_path = out_country / fname_step(k)
            montage.save(out_path, format="PNG")
            print(f"[OK] {out_path}")

    print("[DONE] Written to:", out_base)

if __name__ == "__main__":
    main()