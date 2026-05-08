# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


TARGET_LENGTHS = {1: 21, 2: 20, 3: 19, 4: 18}
NEW_DATES = [
    "11/11/2024", "18/11/2024", "25/11/2024", "02/12/2024", "09/12/2024",
    "16/12/2024", "23/12/2024", "30/12/2024", "06/01/2025", "13/01/2025",
    "20/01/2025", "27/01/2025", "03/02/2025", "10/02/2025", "17/02/2025",
    "24/02/2025", "03/03/2025", "10/03/2025", "17/03/2025", "24/03/2025",
    "31/03/2025"
]

# -----------------------------
# 1. From NPY read IS
# -----------------------------
def load_mean_wis_from_npy(run_dir: Path, step: int) -> float:
    p = run_dir / f"wis80_point_step{step}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Cannot find IS file: {p}")
    return float(np.nanmean(np.load(p)))

def normalize_dates_ddmmyyyy(date_series: pd.Series):
    raw = date_series.astype(str).str.strip()
    parsed = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")

    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(raw.loc[missing], format="%d/%m/%Y", errors="coerce")

    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(raw.loc[missing], dayfirst=True, errors="coerce")

    formatted = parsed.dt.strftime("%d/%m/%Y")
    return formatted.where(parsed.notna(), raw)


def load_clean_df(path: Path):
    df = pd.read_csv(path)

    # date DD/MM/YYYY
    if "date" in df.columns:
        date_series = normalize_dates_ddmmyyyy(df["date"])
    else:
        date_series = None

    # support pred / median / mean / pred_step{N} column names
    pred_col = next((c for c in df.columns if c in ("pred", "median", "mean") or c.startswith("pred_step")), None)
    if pred_col is None:
        raise KeyError("pred")
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    
    true_col = "true" if "true" in df.columns else ("TRUE" if "TRUE" in df.columns else None)
    y_true = pd.to_numeric(df[true_col], errors="coerce") if true_col else pd.Series([np.nan]*len(df))

    lcol = next((c for c in df.columns if "lower" in c.lower() or "0.1" in c or c.lower().startswith("lo")), None)
    ucol = next((c for c in df.columns if "upper" in c.lower() or "0.9" in c or c.lower().startswith("hi")), None)
    
    lower80 = pd.to_numeric(df[lcol], errors="coerce") if lcol else pd.Series([np.nan]*len(df))
    upper80 = pd.to_numeric(df[ucol], errors="coerce") if ucol else pd.Series([np.nan]*len(df))

    return pd.DataFrame({
        "date": date_series,
        "true": y_true,
        "pred": pred,
        "lower80": lower80,
        "upper80": upper80
    })

def align_to_expected_dates(df: pd.DataFrame, expected_dates: list[str], source):
    df = df[df["date"].isin(expected_dates)].copy()
    seen_dates = set(df["date"].dropna().astype(str))
    expected_set = set(expected_dates)
    if seen_dates != expected_set or len(df) != len(expected_dates):
        missing = [d for d in expected_dates if d not in seen_dates]
        extra = sorted(seen_dates - expected_set)
        raise ValueError(
            f"Date alignment failed for {source}: "
            f"expected {len(expected_dates)} rows, got {len(df)}; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    df["date_cat"] = pd.Categorical(df["date"], categories=expected_dates, ordered=True)
    return df.sort_values("date_cat").drop(columns=["date_cat"]).reset_index(drop=True)

# -----------------------------
# 3.get True
# -----------------------------
def get_ref_true(results_root: Path, country: str, target: str, expected_dates: list):
    cands = sorted(list(results_root.rglob("*.csv")), key=lambda p: 0 if "naive" in str(p).lower() else 1)
    
    for p in cands:
        if country.lower() not in str(p).lower() or target.lower() not in str(p).lower():
            continue
        try:
            df = load_clean_df(p)
            if df["true"].notna().any():
                # 强行按要求截取
                df_filtered = df[df["date"].isin(expected_dates)].copy()
                if len(df_filtered) == len(expected_dates):
                    return df_filtered[["date", "true"]]
        except:
            continue
    return None


def weighted_ensemble(dfs: list[pd.DataFrame], mean_wis_list: list[float], expected_dates: list):
    wis = np.array(mean_wis_list, dtype=float)
    wis = np.where(np.isfinite(wis) & (wis > 0), wis, np.inf)
    w = 1.0 / (wis + 1e-12)
    w = w / w.sum()

    n = len(expected_dates)
    pred = np.zeros(n)
    lo = np.zeros(n)
    hi = np.zeros(n)

    for i, df in enumerate(dfs):
        pred += w[i] * df["pred"].to_numpy()
        lo   += w[i] * df["lower80"].to_numpy()
        hi   += w[i] * df["upper80"].to_numpy()

    lo = np.minimum(lo, pred)
    hi = np.maximum(hi, pred)
    lo = np.clip(lo, 0.0, None)

    out = pd.DataFrame({"date": expected_dates, "pred": pred, "lower80": lo, "upper80": hi})
    
    if "true" in dfs[0].columns:
        out.insert(1, "true", dfs[0]["true"].to_numpy())
        
    return out, w


fixed_members = [
    ("ARIMA", "real"),
    ("SEIR", "real"),
    ("TabPFN_ts", "real"),
]

selectable_members = [
    ("DLinear", ["real", "augmented", "combined"]),
    ("LSTM", ["real", "augmented", "combined"]),
    ("Autoformer", ["real", "augmented", "combined"]),
]

def run_one_country(results_root: Path, country: str, target: str):
    out_dir = results_root / f"Ensemble_real_{country}_{target}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in [1, 2, 3, 4]:

        target_len = TARGET_LENGTHS[step]
        expected_dates = NEW_DATES[-target_len:]

        ref_true_df = get_ref_true(results_root, country, target, expected_dates)

        chosen_dfs = []
        chosen_meanwis = []
        chosen_tags = []


        for model, data in fixed_members:
            folder = f"{model}_{data}_{country}_{target}"
            run_dir = results_root / folder

            mean_wis = load_mean_wis_from_npy(run_dir, step)
            fcsv = run_dir / f"rolling_pred_step{step}.csv"
            
            df = load_clean_df(fcsv)

            df = align_to_expected_dates(df, expected_dates, fcsv)

            if ref_true_df is not None:
                df["true"] = ref_true_df["true"].to_numpy()

            chosen_dfs.append(df)
            chosen_meanwis.append(mean_wis)
            chosen_tags.append(folder)


        for model, data_list in selectable_members:
            best = None
            for data in data_list:
                folder = f"{model}_{data}_{country}_{target}"
                run_dir = results_root / folder
                npy_path = run_dir / f"wis80_point_step{step}.npy"
                if not npy_path.exists(): continue
                
                mw = float(np.nanmean(np.load(npy_path)))
                if best is None or mw < best[0]:
                    best = (mw, folder, run_dir)

            if best is None:
                raise FileNotFoundError(f"No WIS files found for {model} {country} {target} step{step}")

            mean_wis, folder, run_dir = best
            fcsv = run_dir / f"rolling_pred_step{step}.csv"
            
            df = load_clean_df(fcsv)

            df = align_to_expected_dates(df, expected_dates, fcsv)

            if ref_true_df is not None:
                df["true"] = ref_true_df["true"].to_numpy()

            chosen_dfs.append(df)
            chosen_meanwis.append(mean_wis)
            chosen_tags.append(folder)


        ens_df, weights = weighted_ensemble(chosen_dfs, chosen_meanwis, expected_dates)
        out_path = out_dir / f"rolling_pred_step{step}.csv"
        ens_df.to_csv(out_path, index=False)

        print(f"\n=== Ensemble {country} {target} step{step} ===")
        print(f"  --> Same length: {target_len} points")
        for tag, mw in zip(chosen_tags, chosen_meanwis):
            print(f"  Model: {tag:35s} | mean IS80={mw:.6f}")
        print("  Weights:", [float(f"{x:.4f}") for x in weights])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument("--target", type=str, default="ILI")
    ap.add_argument("--country", type=str, default=None)
    ap.add_argument("--countries", type=str, default=None)
    ap.add_argument("--all9", action="store_true")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    target = args.target
    preset9 = ["Belgium", "Czechia", "Denmark", "France", "Ireland", "Italy", "Netherlands", "Poland", "Romania"]

    countries = preset9
    if args.countries: countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    elif args.country: countries = [args.country]

    for c in countries:
        print(f"\n\n######## Starting Ensemble calculation for {c} ########")
        try:
            run_one_country(results_root, c, target)
        except Exception as e:
            print(f"[ERROR] {c}: {e}")

if __name__ == "__main__":
    main()
