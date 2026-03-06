import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("results")
OUT_DIR = ROOT / "metrics_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "point_metrics_long_real_sim.csv"

COUNTRIES = [
    "Belgium","Czechia","Denmark","France","Ireland",
    "Italy","Netherlands","Poland","Romania"
]
COUNTRY_LOWER = {c.lower(): c for c in COUNTRIES}

METHODS = ["Naive","ARIMA","DLinear","LSTM","Autoformer","TabPFN_ts","Respicast","Ensemble"]
METHOD_ALIASES = {
    "tabpfn_ts": "TabPFN_ts",
    "tabpfn": "TabPFN_ts",
    "tabpfnts": "TabPFN_ts",
    "autoformer": "Autoformer",
    "dlinear": "DLinear",
    "lstm": "LSTM",
    "naive": "Naive",
    "arima": "ARIMA",
    "Respicast": "Respicast",
    "hubensemble": "Respicast",
    "ensemble":"Ensemble",
}
TRAIN_SETTINGS = ["real","augmented","combined"]
STEPS = [1,2,3,4]

# ---------- metrics ----------
def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def wmape(y, yhat, eps=1e-12):
    denom = np.sum(np.abs(y)) + eps
    return float(np.sum(np.abs(y - yhat)) / denom)

def align_by_pred_start(y_true, y_pred):
    """从 pred 第一个非 NaN 位置开始与 TRUE 对齐"""
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

# ---------- infer helpers ----------
def infer_dataset_type(path: Path) -> str:
    s = str(path).lower()
    if "simulated" in s:
        return "simulated"
    if "real" in s:
        return "real"
    return "real"  # 不明确就跳过，避免误归类

def infer_country_method_trainsetting(path: Path):
    s = str(path).lower()

    country = None
    for c in COUNTRY_LOWER:
        if c in s:
            country = COUNTRY_LOWER[c]
            break

    train_setting = ""
    for t in TRAIN_SETTINGS:
        if re.search(rf"(^|[^a-z]){t}([^a-z]|$)", s):
            train_setting = t
            break

    method = None
    for key, val in METHOD_ALIASES.items():
        if key in s:
            method = val
            break
    if method is None:
        for m in METHODS:
            if m.lower() in s:
                method = m
                break
    if method in ["Naive", "ARIMA", "TabPFN_ts", "Respicast", "Ensemble"]:
        train_setting = ""
    return country, method, train_setting

# ---------- loaders ----------
def load_true_pred_csv(path: Path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    tcol = None
    for k in ["y_true","gt","label","true"]:
        if k in cols:
            tcol = cols[k]; break

    pcol = None
    for k in ["pred","y_pred","forecast","target"]:
        if k in cols:
            pcol = cols[k]; break
    if pcol is None:
        raise ValueError(f"Missing pred/target in {path}")

    y_pred = df[pcol].to_numpy(float)
    y_true = None if tcol is None else df[tcol].to_numpy(float)
    return y_true, y_pred

def load_wis_mean_npy(path: Path) -> float:
    """
    兼容你保存的 npy：
    - 如果是 dict：优先找 wis80 / WIS / value 等
    - 如果是 array：直接 mean
    """
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and isinstance(obj.item(), dict):
        d = obj.item()
        for k in ["wis80", "WIS80", "wis", "WIS", "value", "values", "wis_80"]:
            if k in d:
                arr = np.asarray(d[k], float)
                return float(np.nanmean(arr))
        # fallback: 第一个 array-like
        for v in d.values():
            if isinstance(v, (list, np.ndarray)):
                arr = np.asarray(v, float)
                if arr.ndim >= 1:
                    return float(np.nanmean(arr))
        return float("nan")

    arr = np.asarray(obj, float)
    return float(np.nanmean(arr))



def main():
    # ------- collect point csv files -------
    point_files = []
    for step in STEPS:
        point_files += [(step, p) for p in ROOT.rglob(f"*rolling_pred_step{step}.csv")]
        point_files += [(step, p) for p in ROOT.rglob(f"*tabpfn_ts_pred_step{step}.csv")]

    # ------- build ref TRUE by (dataset_type, step, country) -------
    ref_true = {}  # (dataset_type, step, country) -> y_true
    def preferred(p: Path):
        s = str(p).lower()
        return ("naive" in s) or ("arima" in s)

    for step, p in sorted(point_files, key=lambda x: (0 if preferred(x[1]) else 1, str(x[1]))):
        dt = infer_dataset_type(p)
        if dt == "":
            continue
        country, method, train_setting = infer_country_method_trainsetting(p)
        if country is None:
            continue
        try:
            y_true, y_pred = load_true_pred_csv(p)
            if y_true is not None and (dt, step, country) not in ref_true:
                y_aligned, _, _ = align_by_pred_start(y_true, y_pred)
                if y_aligned is not None:
                    ref_true[(dt, step, country)] = y_aligned
        except Exception:
            pass

    rows = []

    # ------- compute MAE/wMAPE -------
    for step, p in point_files:
        dt = infer_dataset_type(p)
        if dt == "":
            continue

        country, method, train_setting = infer_country_method_trainsetting(p)
        if country is None or method is None:
            continue

        # Respicast 没有 simulated
        if dt == "simulated" and method == "Respicast":
            continue

        try:
            y_true, y_pred = load_true_pred_csv(p)
        except Exception as e:
            print(f"[WARN] cannot read {p}: {e}")
            continue

        if y_true is None:
            y_true = ref_true.get((dt, step, country), None)

        y_aligned, pred_aligned, start_idx = align_by_pred_start(y_true, y_pred)
        if y_aligned is None or pred_aligned is None:
            continue

        mae_v = mae(y_aligned, pred_aligned)
        wm_v = wmape(y_aligned, pred_aligned)

        n_used = len(pred_aligned)

        rows.append({
            "dataset_type": dt,
            "country": country,
            "method": method,
            "train_setting": train_setting,
            "step": step,
            "metric": "MAE",
            "value": mae_v,
            "start_idx": start_idx,
            "n_used": n_used,
            "source_file": str(p),
        })
        rows.append({
            "dataset_type": dt,
            "country": country,
            "method": method,
            "train_setting": train_setting,
            "step": step,
            "metric": "wMAPE",
            "value": wm_v,
            "start_idx": start_idx,
            "n_used": n_used,
            "source_file": str(p),
        })

    # ------- add mean WIS (read from npy) -------
    # 兼容两种命名：wis80_pred_stepK.npy / wis80_point_stepK.npy
    for step in STEPS:
        for pat in [f"*wis80_pred_step{step}.npy", f"*wis80_point_step{step}.npy"]:
            for p in ROOT.rglob(pat):
                dt = infer_dataset_type(p)
                if dt == "":
                    continue
                country, method, train_setting = infer_country_method_trainsetting(p)
                if country is None or method is None:
                    continue
                # Respicast simulated 不会有，但这里也过滤一下
                if dt == "simulated" and method == "Respicast":
                    continue
                try:
                    wis_m = load_wis_mean_npy(p)
                except Exception as e:
                    print(f"[WARN] wis load fail {p}: {e}")
                    continue

                rows.append({
                    "dataset_type": dt,
                    "country": country,
                    "method": method,
                    "train_setting": train_setting,
                    "step": step,
                    "metric": "WIS80_mean",
                    "value": wis_m,
                    "start_idx": None,
                    "n_used": None,
                    "source_file": str(p),
                })

    df_out = pd.DataFrame(rows)

    # ------- dedupe: keep shortest source path for same key -------
    if not df_out.empty:
        df_out["source_len"] = df_out["source_file"].str.len()
        df_out = df_out.sort_values(
            ["dataset_type","country","method","train_setting","step","metric","source_len"]
        )
        df_out = df_out.drop_duplicates(
            ["dataset_type","country","method","train_setting","step","metric"],
            keep="first"
        ).drop(columns=["source_len"])

    df_out.to_csv(OUT_PATH, index=False)
    print(f"[OK] wrote {OUT_PATH} | rows={len(df_out)}")

if __name__ == "__main__":
    main()