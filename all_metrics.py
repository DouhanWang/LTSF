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
TRAIN_SETTINGS = ["real","augmented","combined"]
STEPS = [1,2,3,4]

# ---------- Metrics 函数 ----------
def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def wmape(y, yhat, eps=1e-12):
    denom = np.sum(np.abs(y)) + eps
    return float(np.sum(np.abs(y - yhat)) / denom)

# ---------- 信息提取 Helper ----------
def infer_dataset_type(path: Path) -> str:
    s = str(path).lower()
    return "simulated" if "simulated" in s else "real"

def infer_country_method_trainsetting(path: Path):
    s = str(path).lower()
    country = next((c for k, c in COUNTRY_LOWER.items() if k in s), None)
    train_setting = next((t for t in TRAIN_SETTINGS if re.search(rf"(^|[^a-z]){t}([^a-z]|$)", s)), "")
    
    method = None
    aliases = {
        "tabpfn_ts": "TabPFN_ts", "tabpfn": "TabPFN_ts", "tabpfnts": "TabPFN_ts",
        "autoformer": "Autoformer", "dlinear": "DLinear", "lstm": "LSTM",
        "naive": "Naive", "arima": "ARIMA", "respicast": "Respicast",
        "hubensemble": "Respicast", "ensemble": "Ensemble",
    }
    for key, val in aliases.items():
        if key in s:
            method = val
            break

    if method in ["Naive", "ARIMA", "TabPFN_ts", "Respicast", "Ensemble"]:
        train_setting = ""
        
    return country, method, train_setting

# ---------- 主程序 ----------
def main():
    print("🚀 开始生成统一评估指标 (MAE, wMAPE, WIS80)...")
    point_files = [(step, p) for step in STEPS for p in ROOT.rglob(f"*rolling_pred_step{step}.csv")]

    # 1. 建立以日期为索引的超级真实值字典 (date -> true_value)
    ref_true = {} 
    for step, p in point_files:
        dt = infer_dataset_type(p)
        country, _, _ = infer_country_method_trainsetting(p)
        if country is None: continue
        
        try:
            df = pd.read_csv(p)
            true_col = "true" if "true" in df.columns else ("TRUE" if "TRUE" in df.columns else None)
            if true_col and "date" in df.columns and df[true_col].notna().any():
                # 【改动重点】：把非空的 true 连同 date 一起存成 Series，方便随时通过日期查询
                valid_df = df.dropna(subset=[true_col])
                dates = valid_df["date"].astype(str).str.strip().values
                trues = pd.to_numeric(valid_df[true_col], errors="coerce").values
                ref_true[(dt, step, country)] = pd.Series(index=dates, data=trues)
        except:
            pass

    rows = []

    # 2. 计算点预测指标：MAE 和 wMAPE
    for step, p in point_files:
        dt = infer_dataset_type(p)
        country, method, train_setting = infer_country_method_trainsetting(p)
        if country is None or method is None: continue

        try:
            df = pd.read_csv(p)
            if "date" not in df.columns: continue
            
            dates = df["date"].astype(str).str.strip().values
            y_pred = pd.to_numeric(df["pred"], errors="coerce").values
            
            true_col = "true" if "true" in df.columns else ("TRUE" if "TRUE" in df.columns else None)
            if true_col and df[true_col].notna().any():
                y_true = pd.to_numeric(df[true_col], errors="coerce").values
            else:
                # 【改动重点】：不再检查长度！而是拿着预测数据的 date，去字典里精准匹配 true
                ref_series = ref_true.get((dt, step, country), None)
                if ref_series is not None:
                    y_true = np.array([ref_series.get(d, np.nan) for d in dates])
                else:
                    y_true = np.array([np.nan] * len(y_pred))

            # 剔除掉 NaN (无论是由于没预测还是由于没匹配上日期)
            valid = np.isfinite(y_pred) & np.isfinite(y_true)
            y_t = y_true[valid]
            y_p = y_pred[valid]
            
            # 只要还有有效的配对数据，就计算误差！
            if len(y_t) > 0:
                rows.append({
                    "dataset_type": dt, "country": country, "method": method,
                    "train_setting": train_setting, "step": step,
                    "metric": "MAE", "value": mae(y_t, y_p), 
                    "n_used": len(y_t), "source_file": str(p)
                })
                rows.append({
                    "dataset_type": dt, "country": country, "method": method,
                    "train_setting": train_setting, "step": step,
                    "metric": "wMAPE", "value": wmape(y_t, y_p), 
                    "n_used": len(y_t), "source_file": str(p)
                })
        except Exception as e:
            print(f"[WARN] 处理 CSV 失败 {p}: {e}")

    # 3. 直接读取完美长度的 WIS (.npy)
    for step in STEPS:
        for p in ROOT.rglob(f"*wis80_point_step{step}.npy"):
            dt = infer_dataset_type(p)
            country, method, train_setting = infer_country_method_trainsetting(p)
            if country is None or method is None: continue
            
            try:
                arr = np.load(p)
                wis_m = float(np.nanmean(arr))
                rows.append({
                    "dataset_type": dt, "country": country, "method": method,
                    "train_setting": train_setting, "step": step,
                    "metric": "WIS80_mean", "value": wis_m, 
                    "n_used": len(arr), "source_file": str(p)
                })
            except Exception:
                pass

    # 4. 生成大表并去重
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.drop_duplicates(
            ["dataset_type","country","method","train_setting","step","metric"],
            keep="first"
        )
    
    df_out.to_csv(OUT_PATH, index=False)
    print(f"\n🎉 [OK] 评估大表已生成！")
    print(f"📁 存储位置: {OUT_PATH}")
    print(f"📊 总指标行数: {len(df_out)}")

if __name__ == "__main__":
    main()