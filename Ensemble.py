import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# WIS from npy (already computed)
# -----------------------------
def load_mean_wis_from_npy(run_dir: Path, step: int) -> float:
    p = run_dir / f"wis80_point_step{step}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    arr = np.load(p)
    return float(np.nanmean(arr))


# -----------------------------
# Align (same spirit as all_metrics.py)
# -----------------------------
def align_by_pred_start(y_true, y_pred):
    """Align series starting at the first finite prediction index."""
    y_pred = np.asarray(y_pred, float)
    idx = np.where(np.isfinite(y_pred))[0]
    if len(idx) == 0:
        return None, None, None
    start = int(idx[0])
    y_pred2 = y_pred[start:]

    if y_true is None:
        return None, y_pred2, start

    y_true = np.asarray(y_true, float)
    y_true2 = y_true[start:]
    n = min(len(y_true2), len(y_pred2))
    return y_true2[:n], y_pred2[:n], start


# -----------------------------
# CSV discovery
# -----------------------------
def find_step_file(run_dir: Path, step: int) -> Path | None:
    # support multiple naming patterns (including your TabPFN)
    cands = [
        f"step{step}.csv",
        f"rolling_pred_step{step}.csv",
        f"pred_step{step}.csv",
        f"tabpfn_ts_pred_step{step}.csv",
        f"tabpfn_pred_step{step}.csv",
    ]

    for name in cands:
        p = run_dir / name
        if p.exists():
            return p

    # fallback: case-insensitive lookup
    files = {p.name.lower(): p for p in run_dir.glob("*.csv")}
    for name in cands:
        if name.lower() in files:
            return files[name.lower()]
    return None


# -----------------------------
# Load quantile-style prediction CSV
# Handles:
#   - columns: timestamp, 0.1..0.9, pred (no true)
#   - columns: time/date + lower80/upper80 + pred (+ optional true)
# -----------------------------
def load_pred_quantile_csv(path: Path):
    df = pd.read_csv(path)

    cols = {c.lower(): c for c in df.columns}

    # time
    time_col = None
    for k in ["timestamp", "date", "ds", "time"]:
        if k in cols:
            time_col = cols[k]
            break
    t = None if time_col is None else pd.to_datetime(df[time_col], errors="coerce").to_numpy()

    # optional true
    true_col = None
    for k in ["y_true", "gt", "label", "true", "truth", "ili"]:
        if k in cols:
            true_col = cols[k]
            break
    y_true = None if true_col is None else pd.to_numeric(df[true_col], errors="coerce").to_numpy()

    # pred
    if "pred" in df.columns:
        y_pred = pd.to_numeric(df["pred"], errors="coerce").to_numpy()
    else:
        pcol = None
        for k in ["pred", "y_pred", "forecast", "point", "median"]:
            if k in cols:
                pcol = cols[k]
                break
        if pcol is None:
            raise ValueError(f"Missing pred column in {path}")
        y_pred = pd.to_numeric(df[pcol], errors="coerce").to_numpy()

    # 80% PI: prefer quantiles 0.1/0.9
    if ("0.1" in df.columns) and ("0.9" in df.columns):
        lower80 = pd.to_numeric(df["0.1"], errors="coerce").to_numpy()
        upper80 = pd.to_numeric(df["0.9"], errors="coerce").to_numpy()
    else:
        # ---- robust interval column matching ----
        # candidates by exact-ish keys
        lower_keys = [
            "lower80", "lower_80", "l80",
            "lower80_step1", "lower80_step2", "lower80_step3", "lower80_step4",
            "lower_step1", "lower_step2", "lower_step3", "lower_step4",
            "p10", "q0.1", "quantile_0.1", "lo_0.1", "lo10",
            "lower"
        ]
        upper_keys = [
            "upper80", "upper_80", "u80",
            "upper80_step1", "upper80_step2", "upper80_step3", "upper80_step4",
            "upper_step1", "upper_step2", "upper_step3", "upper_step4",
            "p90", "q0.9", "quantile_0.9", "hi_0.9", "hi90",
            "upper"
        ]

        lcol = None
        ucol = None

        # 1) exact matches (case-insensitive)
        for k in lower_keys:
            if k in cols:
                lcol = cols[k];
                break
        for k in upper_keys:
            if k in cols:
                ucol = cols[k];
                break

        # 2) fallback: regex-like contains match
        if lcol is None:
            for c in df.columns:
                cl = c.lower()
                if ("lower" in cl or cl.startswith("lo")) and ("0.1" in cl or "10" in cl or "p10" in cl):
                    lcol = c;
                    break
        if ucol is None:
            for c in df.columns:
                cl = c.lower()
                if ("upper" in cl or cl.startswith("hi")) and ("0.9" in cl or "90" in cl or "p90" in cl):
                    ucol = c;
                    break

        # 3) final fallback: if there are 0.1-0.9 quantile columns but named like "0.10"
        if lcol is None:
            for c in df.columns:
                if c.strip().lower() in ["0.10", "0.100", "q0.10"]:
                    lcol = c;
                    break
        if ucol is None:
            for c in df.columns:
                if c.strip().lower() in ["0.90", "0.900", "q0.90"]:
                    ucol = c;
                    break

        if lcol is None or ucol is None:
            raise ValueError(f"Missing interval cols in {path}. Columns={list(df.columns)}")

        lower80 = pd.to_numeric(df[lcol], errors="coerce").to_numpy()
        upper80 = pd.to_numeric(df[ucol], errors="coerce").to_numpy()


    return t, y_true, y_pred, lower80, upper80


# -----------------------------
# Build reference TRUE (optional)
# If none exists anywhere, ensemble still works but saves without true
# -----------------------------
def build_ref_true(results_root: Path, country: str, target: str, step: int):
    # try preferred first: naive/arima if present
    cands = []
    cands += list(results_root.rglob(f"*rolling_pred_step{step}.csv"))
    cands += list(results_root.rglob(f"*tabpfn_ts_pred_step{step}.csv"))
    cands += list(results_root.rglob(f"*step{step}.csv"))

    def preferred(p: Path):
        s = str(p).lower()
        return ("naive" in s) or ("arima" in s)

    cands = sorted(cands, key=lambda p: (0 if preferred(p) else 1, str(p)))

    for p in cands:
        s = str(p).lower()
        if country.lower() not in s or target.lower() not in s:
            continue
        try:
            _, y_true, y_pred, _, _ = load_pred_quantile_csv(p)
            if y_true is None:
                continue
            y_aligned, _, _ = align_by_pred_start(y_true, y_pred)
            if y_aligned is not None and len(y_aligned) > 0:
                return y_aligned
        except Exception:
            continue
    return None


# -----------------------------
# Weighted ensemble (1 / meanWIS)
# Works with optional true + optional time
# -----------------------------
def weighted_ensemble(dfs: list[pd.DataFrame], mean_wis_list: list[float]):
    wis = np.array(mean_wis_list, dtype=float)
    wis = np.where(np.isfinite(wis) & (wis > 0), wis, np.inf)
    w = 1.0 / (wis + 1e-12)
    w = w / w.sum()

    use_time = all(("time" in d.columns) for d in dfs)
    if use_time:
        base_cols = ["time"]
        if "true" in dfs[0].columns:
            base_cols.append("true")
        merged = dfs[0][base_cols].copy().sort_values("time").reset_index(drop=True)

        for i, d in enumerate(dfs):
            d2 = d[["time", "pred", "lower80", "upper80"]].copy().sort_values("time").reset_index(drop=True)
            d2 = d2.rename(columns={"pred": f"pred_{i}", "lower80": f"lower_{i}", "upper80": f"upper_{i}"})
            merged = pd.merge(merged, d2, on="time", how="inner")
    else:
        n = min(len(d) for d in dfs)
        merged = pd.DataFrame()
        if "true" in dfs[0].columns:
            merged["true"] = dfs[0]["true"].iloc[:n].to_numpy()
        for i, d in enumerate(dfs):
            merged[f"pred_{i}"] = d["pred"].iloc[:n].to_numpy()
            merged[f"lower_{i}"] = d["lower80"].iloc[:n].to_numpy()
            merged[f"upper_{i}"] = d["upper80"].iloc[:n].to_numpy()

    pred = np.zeros(len(merged))
    lo = np.zeros(len(merged))
    hi = np.zeros(len(merged))

    for i in range(len(dfs)):
        pred += w[i] * merged[f"pred_{i}"].to_numpy()
        lo   += w[i] * merged[f"lower_{i}"].to_numpy()
        hi   += w[i] * merged[f"upper_{i}"].to_numpy()

    lo = np.minimum(lo, pred)
    hi = np.maximum(hi, pred)
    lo = np.clip(lo, 0.0, None)

    out = pd.DataFrame()
    if "time" in merged.columns:
        out["time"] = merged["time"]
    if "true" in merged.columns:
        out["true"] = merged["true"]
    out["pred"] = pred
    out["lower80"] = lo
    out["upper80"] = hi
    return out, w
# ---- ensemble member definitions (GLOBAL) ----
fixed_members = [
    ("ARIMA", "real"),
    ("TabPFN_ts", "real"),
]

selectable_members = [
    ("DLinear", ["real", "augmented", "combined"]),
    ("LSTM", ["real", "augmented", "combined"]),
    ("Autoformer", ["real", "augmented", "combined"]),
]
def run_one_country(results_root: Path, country: str, target: str):
    # 把你现在 main() 里“跑一个国家”的那段代码原封不动搬到这里
    out_dir = results_root / f"ensemble_real_{country}_{target}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in [1, 2, 3, 4]:
        ref_true = build_ref_true(results_root, country, target, step)
        if ref_true is None:
            print(f"[WARN] ref_true not found for {country} {target} step{step}. Saving without true.")

        chosen_dfs = []
        chosen_meanwis = []
        chosen_tags = []

        # --- fixed members ---
        for model, data in fixed_members:
            folder = f"{model}_{data}_{country}_{target}"
            run_dir = results_root / folder

            mean_wis = load_mean_wis_from_npy(run_dir, step)
            fcsv = find_step_file(run_dir, step)
            if fcsv is None:
                raise FileNotFoundError(f"Missing step{step} csv in {run_dir}")

            t, y_true, y_pred, lower80, upper80 = load_pred_quantile_csv(fcsv)
            if y_true is None:
                y_true = ref_true

            y_true_aligned, y_pred_aligned, start = align_by_pred_start(y_true, y_pred)
            _, lower_aligned, _ = align_by_pred_start(None, lower80)
            _, upper_aligned, _ = align_by_pred_start(None, upper80)

            if y_pred_aligned is None:
                raise ValueError(f"All-NaN pred in {fcsv}")

            n = len(y_pred_aligned)
            n = min(n, len(lower_aligned), len(upper_aligned))
            if y_true_aligned is not None:
                n = min(n, len(y_true_aligned))

            df_std = pd.DataFrame({
                "pred": y_pred_aligned[:n],
                "lower80": lower_aligned[:n],
                "upper80": upper_aligned[:n],
            })
            if y_true_aligned is not None:
                df_std["true"] = y_true_aligned[:n]
            if t is not None:
                df_std.insert(0, "time", t[start:start + n])

            chosen_dfs.append(df_std)
            chosen_meanwis.append(mean_wis)
            chosen_tags.append(folder)

        # --- selectable members (pick best by mean WIS from npy) ---
        for model, data_list in selectable_members:
            best = None  # (mean_wis, folder, run_dir)
            for data in data_list:
                folder = f"{model}_{data}_{country}_{target}"
                run_dir = results_root / folder
                npy_path = run_dir / f"wis80_point_step{step}.npy"
                if not npy_path.exists():
                    continue
                mean_wis = float(np.nanmean(np.load(npy_path)))
                if (best is None) or (mean_wis < best[0]):
                    best = (mean_wis, folder, run_dir)

            if best is None:
                raise FileNotFoundError(
                    f"No available npy found for {model} versions={data_list} at step{step} for {country}_{target}"
                )

            mean_wis, folder, run_dir = best
            fcsv = find_step_file(run_dir, step)
            if fcsv is None:
                raise FileNotFoundError(f"Selected {folder} but missing step{step} csv in {run_dir}")

            t, y_true, y_pred, lower80, upper80 = load_pred_quantile_csv(fcsv)
            if y_true is None:
                y_true = ref_true

            y_true_aligned, y_pred_aligned, start = align_by_pred_start(y_true, y_pred)
            _, lower_aligned, _ = align_by_pred_start(None, lower80)
            _, upper_aligned, _ = align_by_pred_start(None, upper80)

            if y_pred_aligned is None:
                raise ValueError(f"All-NaN pred in {fcsv}")

            n = len(y_pred_aligned)
            n = min(n, len(lower_aligned), len(upper_aligned))
            if y_true_aligned is not None:
                n = min(n, len(y_true_aligned))

            df_std = pd.DataFrame({
                "pred": y_pred_aligned[:n],
                "lower80": lower_aligned[:n],
                "upper80": upper_aligned[:n],
            })
            if y_true_aligned is not None:
                df_std["true"] = y_true_aligned[:n]
            if t is not None:
                df_std.insert(0, "time", t[start:start + n])

            chosen_dfs.append(df_std)
            chosen_meanwis.append(mean_wis)
            chosen_tags.append(folder)

        ens_df, weights = weighted_ensemble(chosen_dfs, chosen_meanwis)

        out_path = out_dir / f"rolling_pred_step{step}.csv"
        ens_df.to_csv(out_path, index=False)

        print(f"\n=== Ensemble {country} {target} step{step} ===")
        for tag, mw in zip(chosen_tags, chosen_meanwis):
            print(f"  member: {tag:35s} | meanWIS80={mw:.6f}")
        print("  weights:", [float(f"{x:.4f}") for x in weights])
        print(f"  saved: {out_path}")
    pass
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument("--target", type=str, default="ILI")

    # 新增：支持多国家
    ap.add_argument("--country", type=str, default=None)
    ap.add_argument("--countries", type=str, default=None)  # 逗号分隔
    ap.add_argument("--all9", action="store_true")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    target = args.target

    preset9 = ["Belgium", "Czechia", "Denmark", "France",
               "Ireland", "Italy", "Netherlands", "Poland", "Romania"]

    if args.all9:
        countries = preset9
    elif args.countries:
        countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    elif args.country:
        countries = [args.country]
    else:
        countries = preset9  # 默认也跑 9 个

    for c in countries:
        print(f"\n\n######## Running ensemble for {c} ########")
        try:
            run_one_country(results_root, c, target)
        except Exception as e:
            print(f"[ERROR] {c}: {e}")


if __name__ == "__main__":
    main()
