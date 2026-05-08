# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

#!/usr/bin/env python3
"""
Plot TabPFN-TS rolling forecasts (1-4 step) using the project's utils.tools.visual.

What it does
- Loads TRUE series from:
  - epi4cast/dataset/real_{COUNTRY}_ILI.csv   (preferred when exists)
  - or epi4cast/dataset/simulated_{COUNTRY}_ILI_median.csv
- Takes the LAST 25 points of the series (sorted by date/week).
- Loads rolling prediction CSVs from:
  epi4cast/results/TabPFN_ts_real_{COUNTRY}_ILI/tabpfn_ts_pred_step{1..4}.csv
  (or TabPFN_ts_simulated_{COUNTRY}_ILI_median if you pass --simulated)
- Builds date-aligned Series: true, pred(median), lower(q0.1), upper(q0.9)
- Calls utils.tools.visual(...) to save into:
  epi4cast/test_results/TabPFN_ts_real_{COUNTRY}_ILI/rolling_test_step_{k}.png
  (or epi4cast/test_results/TabPFN_ts_simulated_{COUNTRY}_ILI_median/ ...)

Notes
- Assumes your prediction CSVs have columns:
  timestamp, 0.1, 0.5, 0.9 (plus others)
- The split red line is placed after the first 4 history points (seq_len=4),
  matching your "前四个点是预测之前的点" requirement.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ---- import your visual() ----
# adjust if your package path differs
from utils.tools import visual


def _find_dataset_file(dataset_dir: Path, country: str, simulated: bool) -> Path:
    if simulated:
        p = dataset_dir / f"simulated_{country}_ILI_median.csv"
        if p.exists():
            return p
        raise FileNotFoundError(f"Missing simulated dataset: {p}")
    # real preferred
    p_real = dataset_dir / f"real_{country}_ILI.csv"
    if p_real.exists():
        return p_real
    # fallback: some repos use uppercase/lowercase inconsistently
    p_real2 = dataset_dir / f"real_{country}_ILI".lower()  # unlikely but safe
    if p_real2.exists():
        return p_real2
    # if not found, try simulated
    p_sim = dataset_dir / f"simulated_{country}_ILI_median.csv"
    if p_sim.exists():
        return p_sim
    raise FileNotFoundError(f"Missing dataset. Tried: {p_real}, {p_sim}")


def _to_week_date(df: pd.DataFrame) -> pd.Series:
    """Return a pd.Series datetime for each row."""
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"])
    if "date" in df.columns:
        return pd.to_datetime(df["date"])
    # common in your project: anno + settimana (ISO week Monday)
    if ("anno" in df.columns) and ("settimana" in df.columns):
        week_str = df["settimana"].astype(int).astype(str).str.zfill(2)
        return pd.to_datetime(df["anno"].astype(int).astype(str) + "-W" + week_str + "-1",
                              format="%G-W%V-%u")
    if ("year" in df.columns) and ("week" in df.columns):
        week_str = df["week"].astype(int).astype(str).str.zfill(2)
        return pd.to_datetime(df["year"].astype(int).astype(str) + "-W" + week_str + "-1",
                              format="%G-W%V-%u")
    raise ValueError("Cannot infer dates. Expected one of: timestamp/date or (anno,settimana) or (year,week).")


def _get_target_col(df: pd.DataFrame) -> str:
    # for your ILI datasets, target is usually 'incidenza' (sometimes 'target')
    for c in ["incidenza", "target", "ILI", "value", "y"]:
        if c in df.columns:
            return c
    # last numeric column fallback
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("No numeric target column found.")
    return num_cols[-1]


def load_last25_series(dataset_path: Path) -> pd.Series:
    df = pd.read_csv(dataset_path)
    dates = _to_week_date(df)
    df = df.copy()
    df["__date__"] = dates
    df = df.sort_values("__date__")

    target_col = _get_target_col(df)
    s = pd.Series(df[target_col].to_numpy(dtype=float), index=pd.to_datetime(df["__date__"]), name=target_col)
    s = s.dropna()

    # keep last 25 points
    if len(s) < 25:
        raise ValueError(f"{dataset_path.name}: need >=25 points, got {len(s)}")
    return s.iloc[-25:]


def load_pred_step(pred_csv: Path) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (median, lower, upper) as pd.Series with DatetimeIndex.
    Expected columns: timestamp, 0.1, 0.5, 0.9
    """
    df = pd.read_csv(pred_csv)
    if "timestamp" not in df.columns:
        raise ValueError(f"{pred_csv} missing 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # if multiple item_id, keep 0
    if "item_id" in df.columns:
        df = df[df["item_id"] == 0].copy()

    # pick quantiles (fallback to closest available)
    def pick_q(col: str, fallback: Optional[str] = None) -> str:
        if col in df.columns:
            return col
        if fallback and fallback in df.columns:
            return fallback
        # try float-ish columns (e.g., "0.5")
        candidates = [c for c in df.columns if c.replace(".", "", 1).isdigit()]
        if not candidates:
            raise ValueError(f"{pred_csv} has no quantile columns like '0.5'. Found: {list(df.columns)}")
        # pick nearest
        target = float(col)
        nearest = min(candidates, key=lambda c: abs(float(c) - target))
        return nearest

    c50 = pick_q("0.5")
    c10 = pick_q("0.1")
    c90 = pick_q("0.9")

    idx = pd.to_datetime(df["timestamp"])
    med = pd.Series(df[c50].to_numpy(dtype=float), index=idx, name="pred_median")
    low = pd.Series(df[c10].to_numpy(dtype=float), index=idx, name="pred_q10")
    upp = pd.Series(df[c90].to_numpy(dtype=float), index=idx, name="pred_q90")
    return med, low, upp


def make_aligned_pred(true25: pd.Series,
                      pred_med: pd.Series,
                      pred_low: pd.Series,
                      pred_upp: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Build length-25 series aligned to true25.index with NaNs outside predicted dates.
    """
    idx = true25.index
    pred_full = pd.Series(np.nan, index=idx, name="pred")
    low_full = pd.Series(np.nan, index=idx, name="lower")
    upp_full = pd.Series(np.nan, index=idx, name="upper")

    # intersection by timestamp
    common = idx.intersection(pred_med.index)
    if len(common) == 0:
        # try aligning by position (last N points) as fallback
        n = min(len(pred_med), len(idx))
        common = idx[-n:]
        pred_full.loc[common] = pred_med.iloc[-n:].to_numpy()
        low_full.loc[common] = pred_low.iloc[-n:].to_numpy()
        upp_full.loc[common] = pred_upp.iloc[-n:].to_numpy()
        return pred_full, low_full, upp_full

    pred_full.loc[common] = pred_med.loc[common]
    low_full.loc[common] = pred_low.loc[common]
    upp_full.loc[common] = pred_upp.loc[common]
    return pred_full, low_full, upp_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".",
                    help="Path to repo root containing dataset/ and results/")
    ap.add_argument("--countries", type=str, nargs="+", default=None,
                    help="Country codes/names used in your filenames (e.g., Belgium Italy). If omitted, auto-scan results folders.")
    ap.add_argument("--simulated", action="store_true",
                    help="Only plot simulated_* (otherwise only real_* when countries are provided). When auto-scanning, both real and simulated are plotted.")
    ap.add_argument("--seq_len", type=int, default=4, help="History length for the red split line.")
    ap.add_argument("--last_n", type=int, default=25, help="Number of last points to plot (default 25).")
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    dataset_dir = repo_root / "dataset"
    results_dir = repo_root / "results"
    # User requested: save into test_results/<TabPFN_ts_real_国家_ILI>/rolling_test_step_k.png
    out_root = repo_root / "test_results"

    def parse_country_from_folder(folder_name: str) -> Tuple[str, bool]:
        """Return (country, simulated_flag) from a results folder name."""
        if folder_name.startswith("TabPFN_ts_real_") and folder_name.endswith("_ILI"):
            country = folder_name[len("TabPFN_ts_real_"):-len("_ILI")]
            return country, False
        if folder_name.startswith("TabPFN_ts_simulated_") and folder_name.endswith("_ILI_median"):
            country = folder_name[len("TabPFN_ts_simulated_"):-len("_ILI_median")]
            return country, True
        raise ValueError(f"Unrecognized folder name: {folder_name}")

    # ---- build jobs (country, simulated, pred_folder_name) ----
    jobs: list[Tuple[str, bool, str]] = []
    if args.countries:
        for c in args.countries:
            # When user explicitly passes countries, respect --simulated flag.
            jobs.append((c, bool(args.simulated),
                         f"TabPFN_ts_simulated_{c}_ILI_median" if args.simulated else f"TabPFN_ts_real_{c}_ILI"))
    else:
        # Auto-scan: plot all matching folders under results/
        if results_dir.exists():
            for p in sorted(results_dir.iterdir()):
                if not p.is_dir():
                    continue
                name = p.name
                if name.startswith("TabPFN_ts_real_") and name.endswith("_ILI"):
                    country, sim = parse_country_from_folder(name)
                    jobs.append((country, sim, name))
                elif name.startswith("TabPFN_ts_simulated_") and name.endswith("_ILI_median"):
                    country, sim = parse_country_from_folder(name)
                    jobs.append((country, sim, name))
        if not jobs:
            raise FileNotFoundError(f"No TabPFN results folders found under: {results_dir}")

    for country, simulated_flag, pred_folder_name in jobs:
        # ---- load TRUE (last 25) ----
        ds_path = _find_dataset_file(dataset_dir, country, simulated=simulated_flag)
        true25 = load_last25_series(ds_path)
        if len(true25) != args.last_n:
            true25 = true25.iloc[-args.last_n:]

        # ---- locate prediction folder ----
        pred_folder = results_dir / pred_folder_name
        out_folder = out_root / pred_folder_name
        out_folder.mkdir(parents=True, exist_ok=True)

        for step in [1, 2, 3, 4]:
            pred_csv = pred_folder / f"tabpfn_ts_pred_step{step}.csv"
            if not pred_csv.exists():
                print(f"[WARN] Missing prediction file: {pred_csv} (skip)")
                continue

            med, low, upp = load_pred_step(pred_csv)
            pred_full, low_full, upp_full = make_aligned_pred(true25, med, low, upp)

            out_path = out_folder / f"rolling_test_step_{step}.png"

            visual(
                true=true25,
                preds=pred_full,
                lower=low_full,
                upper=upp_full,
                seq_len=args.seq_len,
                path=str(out_path),
            )
            print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
