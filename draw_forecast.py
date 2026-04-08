import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------------
# 1. 核心配置
# -----------------------------
COUNTRIES = ["Belgium", "Czechia", "Denmark", "France", "Ireland", "Italy", "Netherlands", "Poland", "Romania"]

GRID_LAYOUT = [
    ["ARIMA_real",      "TabPFN_ts_real",       "Respicast_real"],
    ["DLinear_real",    "DLinear_augmented",    "DLinear_combined"],
    ["LSTM_real",       "LSTM_augmented",       "LSTM_combined"],
    ["Autoformer_real", "Autoformer_augmented", "Autoformer_combined"]
]

# 严格按照你要求的预测长度
TARGET_LENGTHS = {1: 21, 2: 20, 3: 19, 4: 18} 

COUNTRY_YLIM = {
    "Belgium": (0, 2000), "Czechia": (0, 450), "Denmark": (0, 500),
    "France": (0, 1200), "Ireland": (0, 200), "Italy": (0, 3000),
    "Netherlands": (0, 300), "Poland": (0, 1800), "Romania": (0, 100),
}

PAPER_COLORS = {
    "ARIMA_real": "#636363",      
    "DLinear_real": "#6BAED6",    "DLinear_augmented": "#3182BD", "DLinear_combined": "#08519C",  
    "LSTM_real": "#74C476",       "LSTM_augmented": "#31A354",   "LSTM_combined": "#006D2C",   
    "Autoformer_real": "#FD8D3C", "Autoformer_augmented": "#E6550D", "Autoformer_combined": "#A63603",  
    "TabPFN_ts_real": "#D81B60",  "Respicast_real": "#B8860B",  "Naive_real": "#9E9E9E",
    "default": "#2B2B2B"
}

def set_paper_style():
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def get_model_style(prefix):
    m_name = prefix.split('_')[0]
    display_names = {"TabPFN": "TabPFN-TS", "Respicast": "RespiCast"}
    m_disp = display_names.get(m_name, m_name)
    
    if "_real" in prefix: label = m_disp
    elif "_augmented" in prefix: label = f"{m_disp} (aug)"
    elif "_combined" in prefix: label = f"{m_disp} (comb)"
    else: label = m_disp
    
    return PAPER_COLORS.get(prefix, PAPER_COLORS["default"]), label

def fetch_ground_truth_series(results_dir: Path, country: str, step: int):
    naive_folder = f"Naive_real_{country}_ILI"
    csv_path = results_dir / naive_folder / f"rolling_pred_step{step}.csv"
    if not csv_path.exists():
        csv_path = results_dir / naive_folder / f"rolling_pred_step_{step}.csv"
    if not csv_path.exists(): return None
    
    try:
        df = pd.read_csv(csv_path)
        date_col = next((c for c in ["date", "time", "timestamp"] if c in df.columns), None)
        true_col = next((c for c in ["true", "TRUE", "incidenza", "target"] if c in df.columns), None)
        if date_col and true_col:
            idx = pd.DatetimeIndex(pd.to_datetime(df[date_col], format='mixed', dayfirst=True))
            return pd.Series(df[true_col].astype(float).to_numpy(), index=idx)
    except: pass
    return None

# -----------------------------
# 2. 绘制大图
# -----------------------------
def plot_montage_for_country_step(results_dir: Path, out_dir: Path, country: str, step: int):
    set_paper_style()
    fig, axes = plt.subplots(4, 3, figsize=(14, 10))
    fig.subplots_adjust(bottom=0.15, hspace=0.15, wspace=0.1)
    
    global_legend_dict = {}
    
    # ========================================================
    # 【核心1】提取雷打不动的 25 点 Master Ground Truth 底座
    # ========================================================
    master_gt_series = fetch_ground_truth_series(results_dir, country, step)
    if master_gt_series is not None:
        master_gt_series = master_gt_series.tail(25)  # 严格锁定 25 个点
        master_dates = master_gt_series.index.to_numpy()
        master_y = master_gt_series.values
    else:
        master_dates = None
        master_y = None

    # 从字典中拿到当前 Step 对应的预测长度 (21/20/19/18)
    pred_len = TARGET_LENGTHS.get(step, 18)

    for r in range(4):
        for c in range(3):
            ax = axes[r, c]
            prefix = GRID_LAYOUT[r][c]
            color, label = get_model_style(prefix)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 

            if c > 0: ax.tick_params(labelleft=False)
            else: ax.set_ylabel("Incidence")
                
            if r < 3: ax.tick_params(labelbottom=False)
            else: plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

            if country in COUNTRY_YLIM:
                ax.set_ylim(COUNTRY_YLIM[country])

            # ========================================================
            # 【核心2】在每个格子里，先把 25 个黑点和 分割线 画好！
            # ========================================================
            if master_dates is not None:
                line_true, = ax.plot(master_dates, master_y, color='black', marker='o', markersize=3.5, linestyle='None', zorder=50)
                if "GroundTruth" not in global_legend_dict:
                    global_legend_dict["GroundTruth"] = line_true
                
                if len(master_dates) > 3:
                    # 将分割线死死钉在日期轴的 第4个点 (index 3) 上
                    ax.axvline(x=master_dates[3], color='gray', linestyle=':', linewidth=1.5, alpha=0.8, zorder=49)

            # ========================================================
            # 【核心3】加载各模型的预测值，严格按字典长度截断
            # ========================================================
            folder_path = results_dir / f"{prefix}_{country}_ILI"
            csv_path = folder_path / f"rolling_pred_step{step}.csv"
            if not csv_path.exists(): csv_path = folder_path / f"rolling_pred_step_{step}.csv"
            
            if not csv_path.exists():
                ax.set_title(f"{label} (Missing)", fontsize=11, color="gray")
                continue
                
            df = pd.read_csv(csv_path)
            date_col = next((col for col in ["date", "time", "timestamp"] if col in df.columns), None)
            if not date_col: continue
            
            df["_date"] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True)
            
            pred_col = next((col for col in ["pred", "0.5", "target"] if col in df.columns), None)
            if not pred_col: continue
            
            # 清除空值后，精准截取尾部 pred_len (21/20/19/18) 个点！
            df_valid = df.dropna(subset=[pred_col])
            df_pred = df_valid.tail(pred_len).copy()
            
            pred_dates = df_pred["_date"].to_numpy()
            y_pred = df_pred[pred_col].values
            
            l_col = next((col for col in df_pred.columns if "lower" in col.lower() or "0.1" in col), None)
            u_col = next((col for col in df_pred.columns if "upper" in col.lower() or "0.9" in col), None)
            y_low = df_pred[l_col].values if l_col else None
            y_up = df_pred[u_col].values if u_col else None

            # 将纯净长度的预测线叠加上去
            line_pred, = ax.plot(pred_dates, y_pred, color=color, linewidth=1.5, zorder=20)
            if label not in global_legend_dict:
                global_legend_dict[label] = line_pred
                
            if y_low is not None and y_up is not None:
                ax.fill_between(pred_dates, y_low, y_up, color=color, alpha=0.2, zorder=10)
                
            ax.text(0.05, 0.9, label, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')

    fig.legend(
        list(global_legend_dict.values()), list(global_legend_dict.keys()),
        loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.02),
        frameon=False, fontsize=11
    )
    
    fig.suptitle(f"Forecast for {country} (Step {step})", y=0.96, fontsize=16, fontweight='bold')
    save_path = out_dir / f"{country}_step{step}_montage.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[OK] Saved {save_path}")

def main():
    repo_root = Path(".").resolve()
    results_dir = repo_root / "results"
    out_dir = repo_root / "test_results" / "montages"
    out_dir.mkdir(parents=True, exist_ok=True)

    for country in COUNTRIES:
        for step in [1, 2, 3, 4]:
            plot_montage_for_country_step(results_dir, out_dir, country, step)

if __name__ == "__main__":
    main()