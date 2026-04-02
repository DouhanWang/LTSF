import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ==========================================
# 绝对真理：完美的日期与长度映射表
# ==========================================
TARGET_LENGTHS = {1: 21, 2: 20, 3: 19, 4: 18}
NEW_DATES = [
    "11/11/2024", "18/11/2024", "25/11/2024", "02/12/2024", "09/12/2024",
    "16/12/2024", "23/12/2024", "30/12/2024", "06/01/2025", "13/01/2025",
    "20/01/2025", "27/01/2025", "03/02/2025", "10/02/2025", "17/02/2025",
    "24/02/2025", "03/03/2025", "10/03/2025", "17/03/2025", "24/03/2025",
    "31/03/2025"
]

# -----------------------------
# 1. 从 NPY 读取 WIS
# -----------------------------
def load_mean_wis_from_npy(run_dir: Path, step: int) -> float:
    p = run_dir / f"wis80_point_step{step}.npy"
    if not p.exists():
        raise FileNotFoundError(f"找不到 WIS 文件: {p}")
    return float(np.nanmean(np.load(p)))

# -----------------------------
# 2. 读取 CSV 并立刻转化为干净的数据字典
# -----------------------------
def load_clean_df(path: Path):
    df = pd.read_csv(path)
    
    # 获取纯字符串日期
    date_series = df["date"].astype(str).str.strip() if "date" in df.columns else None
    
    pred = pd.to_numeric(df["pred"], errors="coerce")
    
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

# -----------------------------
# 3. 提取基准 True
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

# -----------------------------
# 4. 融合逻辑 (无权重等比例平均)
# -----------------------------
def unweighted_ensemble(dfs: list[pd.DataFrame], expected_dates: list):
    n = len(expected_dates)
    num_models = len(dfs)
    
    # 【核心改变】：权重均分
    w = 1.0 / num_models
    weights = [w] * num_models

    pred = np.zeros(n)
    lo = np.zeros(n)
    hi = np.zeros(n)

    # 简单算术平均累加
    for df in dfs:
        pred += w * df["pred"].to_numpy()
        lo   += w * df["lower80"].to_numpy()
        hi   += w * df["upper80"].to_numpy()

    lo = np.minimum(lo, pred)
    hi = np.maximum(hi, pred)
    lo = np.clip(lo, 0.0, None)

    out = pd.DataFrame({"date": expected_dates, "pred": pred, "lower80": lo, "upper80": hi})
    
    if "true" in dfs[0].columns:
        out.insert(1, "true", dfs[0]["true"].to_numpy())
        
    return out, weights

# -----------------------------
# 5. 主程序与成员配置
# -----------------------------
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
    # 输出文件夹名字依然叫 ensemble_real，方便后续评测流程的无缝对接
    out_dir = results_root / f"Ensembleun_real_{country}_{target}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in [1, 2, 3, 4]:
        target_len = TARGET_LENGTHS[step]
        expected_dates = NEW_DATES[-target_len:]

        ref_true_df = get_ref_true(results_root, country, target, expected_dates)

        chosen_dfs = []
        chosen_meanwis = []
        chosen_tags = []

        # --- 固定的成员 ---
        for model, data in fixed_members:
            folder = f"{model}_{data}_{country}_{target}"
            run_dir = results_root / folder

            mean_wis = load_mean_wis_from_npy(run_dir, step)
            fcsv = run_dir / f"rolling_pred_step{step}.csv"
            
            df = load_clean_df(fcsv)
            df = df[df["date"].isin(expected_dates)].copy()
            df["date_cat"] = pd.Categorical(df["date"], categories=expected_dates, ordered=True)
            df = df.sort_values("date_cat").drop(columns=["date_cat"]).reset_index(drop=True)

            if ref_true_df is not None:
                df["true"] = ref_true_df["true"].to_numpy()

            chosen_dfs.append(df)
            chosen_meanwis.append(mean_wis)
            chosen_tags.append(folder)

        # --- 可选的成员 (依然选取表现最好的变体加入等权融合) ---
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

            mean_wis, folder, run_dir = best
            fcsv = run_dir / f"rolling_pred_step{step}.csv"
            
            df = load_clean_df(fcsv)
            df = df[df["date"].isin(expected_dates)].copy()
            df["date_cat"] = pd.Categorical(df["date"], categories=expected_dates, ordered=True)
            df = df.sort_values("date_cat").drop(columns=["date_cat"]).reset_index(drop=True)

            if ref_true_df is not None:
                df["true"] = ref_true_df["true"].to_numpy()

            chosen_dfs.append(df)
            chosen_meanwis.append(mean_wis)
            chosen_tags.append(folder)

        # 【核心修改】：调用无权重的集成方法
        ens_df, weights = unweighted_ensemble(chosen_dfs, expected_dates)
        out_path = out_dir / f"rolling_pred_step{step}.csv"
        ens_df.to_csv(out_path, index=False)

        print(f"\n=== Unweighted Ensemble {country} {target} step{step} ===")
        print(f"  --> 强制对齐长度: {target_len} 个点")
        for tag, mw in zip(chosen_tags, chosen_meanwis):
            print(f"  成员: {tag:35s} | (仅供参考) meanWIS80={mw:.6f}")
        print("  绝对等分权重:", [float(f"{x:.4f}") for x in weights])

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
        print(f"\n\n######## 开始计算 Unweighted Ensemble: {c} ########")
        try:
            run_one_country(results_root, c, target)
        except Exception as e:
            print(f"[ERROR] {c}: {e}")

if __name__ == "__main__":
    main()