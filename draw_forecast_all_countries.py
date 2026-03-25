import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ==========================================
# 1. 颜色与基础配置
# ==========================================
PAPER_COLORS = {
    "ARIMA_real": "#999999",  
    "DLinear_real": "#C6DBEF",  
    "DLinear_aug": "#9ECAE1",
    "DLinear_comb": "#6BAED6",
    "LSTM_real": "#C7E9C0",  
    "LSTM_aug": "#A1D99B",
    "LSTM_comb": "#74C476",
    "Autoformer_real": "#FDD0A2",  
    "Autoformer_aug": "#FDAE6B",
    "Autoformer_comb": "#FD8D3C",
    "TabPFN_ts_real": "#FA9FB5",  
    "ensemble_real": "#BCBDDC",  
    "Respicast_real": "#B2DF8A",  
}

DEFAULT_COUNTRIES = [
    "Belgium", "Czechia", "Denmark", "France", "Ireland",
    "Italy", "Netherlands", "Poland", "Romania",
]

DEFAULT_METHODS = [
    ("ARIMA_real", "ARIMA"),
    ("DLinear_real", "DLinear (real)"),
    ("DLinear_aug", "DLinear (aug)"),  
    ("DLinear_comb", "DLinear (comb)"),
    ("LSTM_real", "LSTM (real)"),
    ("LSTM_aug", "LSTM (aug)"),
    ("LSTM_comb", "LSTM (comb)"),
    ("Autoformer_real", "Autoformer (real)"),
    ("Autoformer_aug", "Autoformer (aug)"),
    ("Autoformer_comb", "Autoformer (comb)"),
    ("TabPFN_ts_real", "TabPFN-TS"),
    ("ensemble_real", "Ensemble"),  
    ("Respicast_real", "RespiCast"),  
]

COUNTRY_YLIM = {
    "Belgium": (0, 2000),
    "Czechia": (0, 500),
    "Denmark": (0, 500),
    "France": (0, 1200),
    "Ireland": (0, 200),
    "Italy": (0, 3000),
    "Netherlands": (0, 250),
    "Poland": (0, 2000),
    "Romania": (0, 100),
}

def get_paper_color(tag):
    return PAPER_COLORS.get(tag, "#CCCCCC")

def _paper_style_rcparams():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

# ==========================================
# 2. 数据读取辅助函数
# ==========================================
def _list_run_dirs(results_root: str):
    return [d for d in glob.glob(os.path.join(results_root, "*")) if os.path.isdir(d)]

def _find_latest_run(run_dirs, country: str, method_tag: str):
    c0 = country.lower()
    tag_lower = method_tag.lower()
    
    if "tabpfn" in tag_lower:
        m_model, m_split = "tabpfn", "real"
    elif "ensemble" in tag_lower:
        m_model, m_split = "ensemble", "real"
    else:
        parts = tag_lower.split('_')
        m_model = parts[0]
        m_split = parts[1] if len(parts) > 1 else "real"
        if m_split == "aug": m_split = "augmented"
        if m_split == "comb": m_split = "combined"

    cand = []
    for d in run_dirs:
        b = os.path.basename(d).lower()
        if c0 in b and m_model in b and m_split in b:
            cand.append(d)
    if not cand:
        return None
    return sorted(cand, key=lambda x: os.path.getmtime(x))[-1]

def _load_data_from_csv(run_dir: str, method_tag: str, horizon: int):
    if "tabpfn" in method_tag.lower():
        filename = f"tabpfn_ts_pred_step{int(horizon)}.csv"
        if not os.path.exists(os.path.join(run_dir, filename)):
            filename = f"tabpfn_ts_pred_step_{int(horizon)}.csv"
    else:
        filename = f"rolling_pred_step{int(horizon)}.csv"
        if not os.path.exists(os.path.join(run_dir, filename)):
            filename = f"rolling_pred_step_{int(horizon)}.csv"

    p = os.path.join(run_dir, filename)
    if not os.path.exists(p):
        return None, None, None, None, None

    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[warn] Failed to read {p}: {e}")
        return None, None, None, None, None

    date_cols = ["date", "dates", "timestamp", "time", "ds", "target_date", "forecast_date"]
    date_col = next((c for c in date_cols if c in df.columns), None)

    true_cols = ["true", "target", "actual", "y", "y_true", "incidenza"]
    true_col = next((c for c in true_cols if c in df.columns), None)

    pred_cols = ["pred", "forecast", "mean", "0.5", "q0.5", "p50", "median", "y_pred"]
    if "Respicast" in method_tag and "target" in df.columns:
        pred_col = "target"
        if true_col == "target":
            true_col = None
    else:
        pred_col = next((c for c in pred_cols if c in df.columns), None)

    low_cols = [f"lower80_step{horizon}", f"lo80_step{horizon}", f"lower_step{horizon}", "lower80", "lower", "lo", "0.1", "q0.1", "p10"]
    up_cols  = [f"upper80_step{horizon}", f"hi80_step{horizon}", f"upper_step{horizon}", "upper80", "upper", "hi", "0.9", "q0.9", "p90"]
    low_col = next((c for c in low_cols if c in df.columns), None)
    up_col  = next((c for c in up_cols if c in df.columns), None)

    if not pred_col:
        return None, None, None, None, None

    # 直接按字符串提取日期，不进行 pd.to_datetime 解析
    dates = df[date_col].fillna("").astype(str).values if date_col else None
    
    pred_vals = pd.to_numeric(df[pred_col], errors='coerce').values
    true_vals = pd.to_numeric(df[true_col], errors='coerce').values if true_col else None
    lower_vals = pd.to_numeric(df[low_col], errors='coerce').values if low_col else None
    upper_vals = pd.to_numeric(df[up_col], errors='coerce').values if up_col else None

    # 保留最后 25 个点 (严格按倒数截取对齐)
    last_n = 25
    if len(pred_vals) > last_n:
        if dates is not None: dates = dates[-last_n:]
        pred_vals = pred_vals[-last_n:]
        if true_vals is not None: true_vals = true_vals[-last_n:]
        if lower_vals is not None: lower_vals = lower_vals[-last_n:]
        if upper_vals is not None: upper_vals = upper_vals[-last_n:]

    return pred_vals, true_vals, dates, lower_vals, upper_vals

# ==========================================
# 3. 核心绘图逻辑 3x3
# ==========================================
def plot_combined_forecast_3x3(
        results_root="./results",
        horizon=1,
        seq_len=4,  
        out_path="./test_results/combined_forecast_step1.png",
):
    _paper_style_rcparams()
    
    run_dirs = _list_run_dirs(results_root)
    if not run_dirs:
        print(f"Error: No results found in {results_root}")
        return

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.ravel()
    global_handles_dict = {}

    for idx, country in enumerate(DEFAULT_COUNTRIES):
        ax = axes[idx]
        true_plotted = False
        master_dates = None  # 用于记录这幅图的 x 轴日期标签
        
        for method_tag, method_label in DEFAULT_METHODS:
            m_dir = _find_latest_run(run_dirs, country, method_tag)
            if m_dir is None:
                continue
                
            m_pred, m_true, m_dates, m_lower, m_upper = _load_data_from_csv(m_dir, method_tag, horizon)
            if m_pred is None:
                continue

            # 抓取第一组有效的日期字符串供 X 轴使用
            if master_dates is None and m_dates is not None and len(m_dates) > 0:
                master_dates = m_dates
            
            # 利用 np.arange 强行作为 X 轴坐标
            x_pred = np.arange(len(m_pred))
            
            # 1. 绘制 True Data (黑色散点，大小为3) 和红色分界线
            if not true_plotted and m_true is not None and not np.isnan(m_true).all():
                x_true = np.arange(len(m_true))
                line_true, = ax.plot(x_true, m_true, label="True Data", 
                                     color="black", marker='o', markersize=3, alpha=0.8, linestyle="None", zorder=10)
                
                if "True Data" not in global_handles_dict:
                    global_handles_dict["True Data"] = line_true
                
                if len(m_pred) > seq_len:
                    line_div = ax.axvline(x=seq_len, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)
                    if "Forecast Start" not in global_handles_dict:
                        global_handles_dict["Forecast Start"] = line_div
                        
                true_plotted = True
            
            color = get_paper_color(method_tag)
            
            # 2. 绘制预测折线 (全部统一使用实线)
            line_pred, = ax.plot(x_pred, m_pred, label=method_label, 
                                 color=color, linewidth=1.5, linestyle="-", alpha=0.9, zorder=5)

            # 3. 绘制预测区间带
            if m_lower is not None and m_upper is not None:
                ax.fill_between(x_pred, m_lower, m_upper, color=color, alpha=0.15, zorder=4)
            
            if method_label not in global_handles_dict:
                global_handles_dict[method_label] = line_pred

        # 手动将日期字符串覆盖到 X 轴上
        if master_dates is not None:
            n_dates = len(master_dates)
            # 均匀挑选最多 5 个点显示在 X 轴，防止拥挤
            tick_idx = np.linspace(0, n_dates - 1, min(5, n_dates), dtype=int)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([master_dates[i] for i in tick_idx], rotation=30, ha="right")

        # 样式设置
        ax.set_title(country, loc="left", fontsize=13, fontweight="bold")
        if country in COUNTRY_YLIM:
            ax.set_ylim(COUNTRY_YLIM[country])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        r, c = divmod(idx, 3)
        if c == 0:
            ax.set_ylabel("Incidence", fontsize=11)
        if r == 2:
            ax.set_xlabel("Date", fontsize=11)

    # 统一提取 Legend
    handles = []
    labels = []
    
    for fixed_label in ["True Data", "Forecast Start"]:
        if fixed_label in global_handles_dict:
            handles.append(global_handles_dict[fixed_label])
            labels.append(fixed_label)
        
    for _, method_label in DEFAULT_METHODS:
        if method_label in global_handles_dict:
            handles.append(global_handles_dict[method_label])
            labels.append(method_label)

    # 增加底部空间放置图例
    fig.subplots_adjust(wspace=0.2, hspace=0.35, bottom=0.15)
    
    fig.legend(handles, labels, 
               loc="lower center", 
               ncol=7, 
               bbox_to_anchor=(0.5, 0.02),
               fontsize=11,
               frameon=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined forecast grid to: {out_path}")

if __name__ == "__main__":
    plot_combined_forecast_3x3(
        results_root="./results",
        horizon=4,
        seq_len=4,
        out_path="./test_results/montages/all_countries_forecast_step4.png"
    )