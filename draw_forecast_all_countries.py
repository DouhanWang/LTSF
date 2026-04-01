import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines

# ==========================================
# 1. 颜色与基础配置
# ==========================================
PAPER_COLORS = {
    "ARIMA_real": "#636363", 
    "DLinear_real": "#6BAED6",  
    "DLinear_aug": "#3182BD",   
    "DLinear_comb": "#08519C",  
    "LSTM_real": "#74C476",     
    "LSTM_aug": "#31A354",      
    "LSTM_comb": "#006D2C",     
    "Autoformer_real": "#FD8D3C", 
    "Autoformer_aug": "#E6550D",  
    "Autoformer_comb": "#A63603", 
    "TabPFN_ts_real": "#D81B60",  
    "Respicast_real": "#1B9E77",  
    "ensemble_real": "#7570B3",   
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
# 2. 数据读取与误差计算
# ==========================================
def calc_mae(pred, true):
    valid = ~np.isnan(pred) & ~np.isnan(true)
    if not np.any(valid): return np.inf
    return np.mean(np.abs(pred[valid] - true[valid]))

def calc_wis80(pred, true, lower, upper):
    valid = ~np.isnan(pred) & ~np.isnan(true) & ~np.isnan(lower) & ~np.isnan(upper)
    if not np.any(valid): return np.inf
    
    y = true[valid]
    m = pred[valid]
    l = lower[valid]
    u = upper[valid]
    
    alpha = 0.2 # 80% coverage
    
    ae = np.abs(y - m)
    disp = u - l
    under = (2 / alpha) * (l - y) * (y < l)
    over = (2 / alpha) * (y - u) * (y > u)
    
    is80 = disp + under + over
    wis = 0.5 * ae + (alpha / 2) * is80
    return np.mean(wis)

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
    if not cand: return None
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
    if not os.path.exists(p): return None, None, None, None, None

    try:
        df = pd.read_csv(p)
    except Exception as e:
        return None, None, None, None, None

    date_col = next((c for c in ["date", "dates", "timestamp", "time", "ds", "target_date", "forecast_date"] if c in df.columns), None)
    true_col = next((c for c in ["true", "target", "actual", "y", "y_true", "incidenza"] if c in df.columns), None)

    if "Respicast" in method_tag and "target" in df.columns:
        pred_col = "target"
        if true_col == "target": true_col = None
    else:
        pred_col = next((c for c in ["pred", "forecast", "mean", "0.5", "q0.5", "p50", "median", "y_pred"] if c in df.columns), None)

    low_col = next((c for c in [f"lower80_step{horizon}", f"lo80_step{horizon}", f"lower_step{horizon}", "lower80", "lower", "lo", "0.1", "q0.1", "p10"] if c in df.columns), None)
    up_col  = next((c for c in [f"upper80_step{horizon}", f"hi80_step{horizon}", f"upper_step{horizon}", "upper80", "upper", "hi", "0.9", "q0.9", "p90"] if c in df.columns), None)

    if not pred_col: return None, None, None, None, None

    dates = df[date_col].fillna("").astype(str).values if date_col else None
    pred_vals = pd.to_numeric(df[pred_col], errors='coerce').values
    true_vals = pd.to_numeric(df[true_col], errors='coerce').values if true_col else None
    lower_vals = pd.to_numeric(df[low_col], errors='coerce').values if low_col else None
    upper_vals = pd.to_numeric(df[up_col], errors='coerce').values if up_col else None

    # 返回原数据，交由外部统一截取
    return pred_vals, true_vals, dates, lower_vals, upper_vals

# ==========================================
# 3. 核心绘图逻辑 3x3
# ==========================================
def plot_combined_forecast_3x3(
        results_root="./results",
        horizon=4,
        metric_type="MAE", 
        out_path="./test_results/combined_forecast_step4.png",
):
    _paper_style_rcparams()
    
    run_dirs = _list_run_dirs(results_root)
    if not run_dirs:
        print(f"Error: No results found in {results_root}")
        return

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.ravel()

    for idx, country in enumerate(DEFAULT_COUNTRIES):
        ax = axes[idx]
        raw_country_data = {}
        
        # 3.1 预加载该国家所有模型的数据
        for method_tag, method_label in DEFAULT_METHODS:
            m_dir = _find_latest_run(run_dirs, country, method_tag)
            if m_dir is None: continue
                
            m_pred, m_true, m_dates, m_lower, m_upper = _load_data_from_csv(m_dir, method_tag, horizon)
            if m_pred is None: continue

            raw_country_data[method_tag] = {
                "label": method_label,
                "pred": m_pred, "true": m_true, "dates": m_dates,
                "lower": m_lower, "upper": m_upper
            }

        if not raw_country_data:
            ax.set_title(f"{country} (No Data)", loc="left")
            continue

        # ========================================================
        # 3.2 严格按照设定截取：True 24 个点，Pred 17 个点
        # ========================================================
        master_true = None
        master_dates = None
        
        # 寻找全局统一的 Ground Truth 和 Dates（取最后 24 个点）
        for data in raw_country_data.values():
            if master_true is None and data["true"] is not None and len(data["true"]) >= 24:
                master_true = data["true"][-24:]
            if master_dates is None and data["dates"] is not None and len(data["dates"]) >= 24:
                master_dates = data["dates"][-24:]
        
        # 如果由于某种原因没有 24 个点，就取所有可用的点作为容错
        if master_true is None:
            first_data = list(raw_country_data.values())[0]
            master_true = first_data["true"][-24:] if first_data["true"] is not None else np.zeros(24)
            master_dates = first_data["dates"][-24:] if first_data["dates"] is not None else ["" for _ in range(24)]

        # 整理每个模型的预测数据（严格取最后 17 个点）
        country_data = {}
        for tag, data in raw_country_data.items():
            m_pred = data["pred"][-17:] if data["pred"] is not None else None
            m_lower = data["lower"][-17:] if data["lower"] is not None else None
            m_upper = data["upper"][-17:] if data["upper"] is not None else None
            
            # 为了公平计算误差，截取 master_true 的最后 17 个点来与预测值对齐
            m_true_for_metric = master_true[-17:]
            
            if metric_type == "WIS":
                metric_val = calc_wis80(m_pred, m_true_for_metric, m_lower, m_upper)
            else:
                metric_val = calc_mae(m_pred, m_true_for_metric)
                
            country_data[tag] = {
                "label": data["label"],
                "pred": m_pred, "lower": m_lower, "upper": m_upper,
                "metric": metric_val
            }

        # 3.3 排序与透明度映射 (最好 0.95，最差 0.15)
        sorted_tags = sorted(country_data.keys(), key=lambda x: country_data[x]["metric"])
        num_methods = len(sorted_tags)
        alphas_map = {tag: a for tag, a in zip(sorted_tags, np.linspace(0.95, 0.15, num_methods))}

        # ==========================================
        # 3.4 绘制全局 Ground Truth 与分界线
        # ==========================================
        # X 轴坐标: Ground truth 对应 0 到 23
        x_true = np.arange(len(master_true))
        ax.plot(x_true, master_true, label="True Data", color="black", marker='o', markersize=3.5, alpha=0.9, linestyle="None", zorder=50)
        
        # "在 ground truth 的第五个点画分界线" (索引 0, 1, 2, 3, 4 -> 第五点是 x=4)
        split_idx = 4
        if len(master_true) > split_idx:
            ax.axvline(x=split_idx, color='gray', linestyle=':', linewidth=1.5, alpha=0.8, zorder=49)
        
        # ==========================================
        # 3.5 逆序遍历绘制预测值 (确保好模型在最上层)
        # ==========================================
        for rank, tag in enumerate(reversed(sorted_tags)):
            data = country_data[tag]
            alpha_val = alphas_map[tag]
            color = get_paper_color(tag)
            
            # 预测值 X 坐标定位：因为总长 24，预测长 17，所以起点是 24-17 = 7
            start_x = len(master_true) - len(data["pred"])
            x_pred = np.arange(start_x, len(master_true))
            
            z_idx = 10 + rank 
            lw = 1.0 + (alpha_val * 1.5) 
            
            # 绘制均值线
            ax.plot(x_pred, data["pred"], color=color, linewidth=lw, linestyle="-", alpha=alpha_val, zorder=z_idx)

            # 绘制 80% 预测区间
            if data["lower"] is not None and data["upper"] is not None:
                fill_alpha = min(alpha_val * 0.25, 0.25)
                ax.fill_between(x_pred, data["lower"], data["upper"], color=color, alpha=fill_alpha, zorder=z_idx-1)

        # ==========================================
        # 3.6 格式化 X 轴日期
        # ==========================================
        if master_dates is not None:
            n_dates = len(master_dates)
            tick_idx = np.linspace(0, n_dates - 1, min(6, n_dates), dtype=int)
            ax.set_xticks(tick_idx)
            
            # 截取 YYYY-MM 格式
            formatted_dates = []
            for i in tick_idx:
                d_str = master_dates[i]
                if len(d_str) >= 7 and "-" in d_str:
                    formatted_dates.append(d_str[:7])
                else:
                    formatted_dates.append(d_str)
            
            ax.set_xticklabels(formatted_dates, rotation=25, ha="right")

        # 子图美化
        ax.set_title(f"{country}", loc="left", fontsize=13, fontweight="bold")
        if country in COUNTRY_YLIM: ax.set_ylim(COUNTRY_YLIM[country])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        r, c = divmod(idx, 3)
        if c == 0: ax.set_ylabel("Incidence", fontsize=11)
        if r == 2: ax.set_xlabel("Date", fontsize=11)

    # 3.7 全局图例
    legend_elements = [
        mlines.Line2D([], [], color='black', marker='o', markersize=4, linestyle='None', label='True Data'),
        mlines.Line2D([], [], color='gray', linestyle=':', linewidth=1.5, label='Split')
    ]
    for tag, label in DEFAULT_METHODS:
        legend_elements.append(mlines.Line2D([], [], color=get_paper_color(tag), linewidth=2.5, label=label))

    fig.subplots_adjust(wspace=0.2, hspace=0.35, bottom=0.15)
    fig.legend(handles=legend_elements, loc="lower center", ncol=8, bbox_to_anchor=(0.5, 0.02), fontsize=10, frameon=False)
    fig.suptitle(f"Forecast Horizon {horizon} (Sorted by {metric_type}: High Opacity = Better Performance)", fontsize=15, fontweight="bold", y=0.93)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {metric_type} ranked forecast grid to: {out_path}")

if __name__ == "__main__":
    plot_combined_forecast_3x3(
        results_root="./results",
        horizon=4,
        metric_type="MAE",
        out_path="./test_results/montages/all_countries_forecast_step4_MAE.png"
    )

    plot_combined_forecast_3x3(
        results_root="./results",
        horizon=4,
        metric_type="WIS",
        out_path="./test_results/montages/all_countries_forecast_step4_WIS.png"
    )