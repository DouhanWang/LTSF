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

# 把 Ensembleun 放在 Ensemble 前面！
METHODS = ["Naive", "ARIMA", "DLinear", "LSTM", "Autoformer", "TabPFN_ts", "Respicast", "Ensembleun", "Ensemble"]
TRAIN_SETTINGS = ["real", "augmented", "combined"]
STEPS = [1, 2, 3, 4]

# ==========================================
# 强制时间窗口标尺 (保障评估绝对公平)
# ==========================================
NEW_DATES = [
    "11/11/2024", "18/11/2024", "25/11/2024", "02/12/2024", "09/12/2024",
    "16/12/2024", "23/12/2024", "30/12/2024", "06/01/2025", "13/01/2025",
    "20/01/2025", "27/01/2025", "03/02/2025", "10/02/2025", "17/02/2025",
    "24/02/2025", "03/03/2025", "10/03/2025", "17/03/2025", "24/03/2025",
    "31/03/2025"
]
TARGET_LENGTHS = {1: 21, 2: 20, 3: 19, 4: 18}

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
    method = next((m for m in METHODS if m.lower() in s), None)
    train_setting = ""
    if method in ["DLinear", "LSTM", "Autoformer"]:
        train_setting = next((t for t in TRAIN_SETTINGS if re.search(rf"(^|[^a-z]){t}([^a-z]|$)", s)), "")
    return country, method, train_setting

# ---------- 主程序 ----------
def main():
    print("🚀 开始生成绝对公平的统一评估指标 (MAE, wMAPE, WIS80)...")
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

        # 【核心拦截】：获取当前 Step 必须评估的合法日期！
        expected_len = TARGET_LENGTHS.get(step, 21)
        expected_dates = set(NEW_DATES[-expected_len:])

        try:
            df = pd.read_csv(p)
            if "date" not in df.columns: continue
            
            # 强行过滤！只保留在 expected_dates 范围内的预测数据
            df["date_str"] = df["date"].astype(str).str.strip()
            df = df[df["date_str"].isin(expected_dates)]
            
            dates = df["date_str"].values
            y_pred = pd.to_numeric(df["pred"], errors="coerce").values
            
            true_col = "true" if "true" in df.columns else ("TRUE" if "TRUE" in df.columns else None)
            if true_col and df[true_col].notna().any():
                y_true = pd.to_numeric(df[true_col], errors="coerce").values
            else:
                ref_series = ref_true.get((dt, step, country), None)
                if ref_series is not None:
                    y_true = np.array([ref_series.get(d, np.nan) for d in dates])
                else:
                    y_true = np.array([np.nan] * len(y_pred))

            valid = np.isfinite(y_pred) & np.isfinite(y_true)
            y_t = y_true[valid]
            y_p = y_pred[valid]
            
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

    # 3. 读取 WIS (.npy)
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

if __name__ == "__main__":
    main()