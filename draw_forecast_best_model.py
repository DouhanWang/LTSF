# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

# ==========================================
# 1. Color and setting definitions
# ==========================================
PAPER_COLORS = {
    "ARIMA_real": "#636363",
    "SEIR_real": "#008B8B",
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
    "Respicast_real": "#B8860B",
    "ensemble_real": "#7570B3",   
}

DEFAULT_COUNTRIES = [
    "Belgium", "Czechia", "Denmark", "France", "Ireland",
    "Italy", "Netherlands", "Poland", "Romania",
]

EXCLUDED_BEST_METHODS = {
    "ensembleun",
    "ensemble_un",
    "ensemble-un",
    "unweighted_ensemble",
    "unweighted ensemble",
}

COUNTRY_YLIM = {
    "Belgium": (0, 1600),
    "Czechia": (0, 350),
    "Denmark": (0, 400),
    "France": (0, 800),
    "Ireland": (0, 150),
    "Italy": (0, 2500),
    "Netherlands": (0, 200),
    "Poland": (0, 1000),
    "Romania": (0, 70),
}

TARGET_LENGTHS = {1: 21, 2: 20, 3: 19, 4: 18}

def _paper_style_rcparams():
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 12.5, "axes.labelsize": 12.5,
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "axes.linewidth": 1.0,
    })

def get_model_identity(method, setting):
    m = str(method).strip().lower()
    s = str(setting).strip().lower() if pd.notna(setting) else ""
    
    if m == "tabpfn_ts": return "TabPFN-TS", "TabPFN_ts_real"
    if m == "ensemble": return "Ensemble", "ensemble_real"
    if m == "respicast": return "RespiCast", "Respicast_real"
    if m == "naive": return "Naive", "ARIMA_real"
    if m == "arima": return "ARIMA", "ARIMA_real"
    if m == "seir": return "SEIR", "SEIR_real"
    
    if m == "dlinear": m_display = "DLinear"
    elif m == "lstm": m_display = "LSTM"
    elif m == "autoformer": m_display = "Autoformer"
    else: m_display = str(method)
    
    if s == "augmented":
        s_display, s_tag = "aug", "aug"
    elif s == "combined":
        s_display, s_tag = "comb", "comb"
    else:
        s_display, s_tag = "real", "real"
        
    label = f"{m_display} ({s_display})"
    color_key = f"{m_display}_{s_tag}"
    return label, color_key

# ==========================================
# 2. read data
# ==========================================
def get_csv_path_from_row(row, horizon):
    """【read rolling_pred_stepX.csv"""
    source_path = Path(str(row["source_file"]).replace("\\", "/"))
    run_dir = source_path.parent  # only folder directory
    
    csv_path = run_dir / f"rolling_pred_step{int(horizon)}.csv"
    
    if not csv_path.exists(): 
        csv_path = run_dir / f"rolling_pred_step_{int(horizon)}.csv"
        
    return csv_path

def load_true_from_csv(csv_path, last_n=25):
    p = Path(str(csv_path).replace("\\", "/"))
    if not p.exists(): return None, None
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    
    tcol = next((cols[k] for k in ["y_true", "true", "gt", "incidenza"] if k in cols), None)
    dcol = next((cols[k] for k in ["date", "dates", "timestamp", "target_date"] if k in cols), None)
    
    if not tcol: return None, None
    
    t_vals = pd.to_numeric(df[tcol], errors='coerce').values
    d_vals = df[dcol].astype(str).values if dcol else np.array(["" for _ in range(len(df))])
    
    valids = np.where(~np.isnan(t_vals))[0]
    if len(valids) == 0: return None, None
    end_idx = valids[-1] + 1  
    
    t_vals = t_vals[:end_idx]
    d_vals = d_vals[:end_idx]
    
    return t_vals[-last_n:], d_vals[-last_n:]

def load_pred_from_csv(csv_path, last_n=18):
    p = Path(str(csv_path).replace("\\", "/"))
    if not p.exists(): return None, None, None

    df = pd.read_csv(p)

    cols = {c.lower(): c for c in df.columns}
    
    pcol = next((cols[k] for k in ["pred", "y_pred", "forecast", "target", "0.5", "median"] if k in cols), None)
    if not pcol: return None, None, None
    
    p_vals = pd.to_numeric(df[pcol], errors='coerce').values

    lcol = next((cols[k] for k in cols if "lower" in k or "lo" in k or "0.1" in k), None)
    ucol = next((cols[k] for k in cols if "upper" in k or "hi" in k or "0.9" in k), None)
    
    l_vals = pd.to_numeric(df[lcol], errors='coerce').values if lcol else None
    u_vals = pd.to_numeric(df[ucol], errors='coerce').values if ucol else None
    
    valids = np.where(~np.isnan(p_vals))[0]
    if len(valids) == 0: return None, None, None
    end_idx = valids[-1] + 1 
    
    p_vals = p_vals[:end_idx]

    if l_vals is not None: l_vals = l_vals[:end_idx]
    if u_vals is not None: u_vals = u_vals[:end_idx]
    
    def _slice(arr):
        if arr is None: return None
        if len(arr) >= last_n: return arr[-last_n:]
        pad = np.full(last_n - len(arr), np.nan)
        return np.concatenate([pad, arr])
        
    return _slice(p_vals), _slice(l_vals), _slice(u_vals)

# ==========================================
# 3. Plotting function
# ==========================================
def plot_point_forecast_with_interval(
        metrics_csv="./results/metrics_tables/point_metrics_long_real_sim.csv",
        horizon=4,
        metric_type="MAE",
        out_path="./test_results/montages/all_countries_forecast_step4_MAE_Interval.png"
):
    _paper_style_rcparams()
    
    if not os.path.exists(metrics_csv):
        print(f"Error: Cannot find {metrics_csv}")
        return

    df_metrics = pd.read_csv(metrics_csv)
    df_sub = df_metrics[
        (df_metrics["dataset_type"] == "real") &
        (df_metrics["step"] == horizon) &
        (df_metrics["metric"] == metric_type)  
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.ravel()
    plotted_global_labels = {}

    for idx, country in enumerate(DEFAULT_COUNTRIES):
        ax = axes[idx]
        df_c = df_sub[df_sub["country"].str.lower() == country.lower()]
        
        if df_c.empty:
            ax.set_title(f"{country} (No Data)", loc="left")
            continue

        # ----------------------------------------------------
        # 1. Get Ground Truth (Use Naive )
        # ----------------------------------------------------
        naive_row = df_c[df_c["method"].str.lower() == "naive"]
        if naive_row.empty:
            ax.set_title(f"{country} (No Naive GT)", loc="left")
            continue

        pred_len = TARGET_LENGTHS.get(int(horizon), 18)
        gt_path = get_csv_path_from_row(naive_row.iloc[0], horizon)
        plot_true, plot_dates = load_true_from_csv(gt_path, last_n=25)
        
        if plot_true is None:
            ax.set_title(f"{country} (GT Read Failed)", loc="left")
            continue
            
        len_t = len(plot_true)

        x_true = np.arange(len_t)
        
        ax.plot(x_true, plot_true, color="black", marker='o', markersize=3.5, alpha=0.9, linestyle="None", zorder=50)

        ax.axvline(x=3, color='gray', linestyle=':', linewidth=1.5, alpha=0.8, zorder=49)

        # ----------------------------------------------------
        # 2. Find best model
        # ----------------------------------------------------
        method_key = df_c["method"].astype(str).str.strip().str.lower()
        respi_row = df_c[method_key == "respicast"]
        others = df_c[(method_key != "respicast") & (~method_key.isin(EXCLUDED_BEST_METHODS))]
        
        rows_to_plot = []
        if not respi_row.empty:
            rows_to_plot.append(respi_row.iloc[0])
            
        if not others.empty:
            best_row = others.loc[others["value"].idxmin()]
            rows_to_plot.append(best_row)
            m_set_str = best_row['train_setting'] if pd.notna(best_row['train_setting']) else ''
            metric_display = "IS" if metric_type == "WIS80_mean" else metric_type
            print(f"[{country}] 查表锁定 {metric_display} 最强: {best_row['method']} {m_set_str} ({metric_display}: {best_row['value']:.2f})")


        for i, row in enumerate(rows_to_plot):
            m_name = row["method"]
            m_set  = row["train_setting"]
            
            csv_path = get_csv_path_from_row(row, horizon)
            
            p_17, l_17, u_17 = load_pred_from_csv(csv_path, last_n=18)
            if p_17 is None:
                continue
                
            label, color_key = get_model_identity(m_name, m_set)
            color = PAPER_COLORS.get(color_key, "#333333")
            
            len_p = len(p_17)
            x_pred = np.arange(len_t - len_p, len_t)
            
            ax.plot(x_pred, p_17, color=color, linewidth=2.0, linestyle="-", alpha=0.9, zorder=20)
            if l_17 is not None and u_17 is not None:
                ax.fill_between(x_pred, l_17, u_17, color=color, alpha=0.2, zorder=10)

            plotted_global_labels[label] = color


        tick_idx = np.linspace(0, len_t - 1, min(6, len_t), dtype=int)
        ax.set_xticks(tick_idx)
        
        formatted_dates = []
        for i in tick_idx:
            raw_str = str(plot_dates[i])
            try:
                dt = pd.to_datetime(raw_str, format='mixed', dayfirst=True)
                formatted_dates.append(dt.strftime("%Y-%m"))
            except:
                formatted_dates.append(raw_str[:7])
                
        ax.set_xlim(-0.5, len_t - 0.5)
        ax.set_title(f"{country}", loc="left", fontsize=13, fontweight="bold")
        if country in COUNTRY_YLIM: ax.set_ylim(COUNTRY_YLIM[country])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        r, c = divmod(idx, 3)
        if c == 0: 
            ax.set_ylabel("Incidence", fontsize=13)
            
        # Show only on the last row (r == 2) X-axis labels and Date title
        if r == 2: 
            ax.set_xticklabels(formatted_dates, rotation=25, ha="right")
            ax.set_xlabel("Date", fontsize=13)
        else:
            ax.set_xticklabels([])  # 隐藏其他行的文字标签
            ax.tick_params(axis='x', bottom=True, labelbottom=False) # 保留短刻度线，但不显示文字

    # ----------------------------------------------------
    # 5. Legend
    # ----------------------------------------------------
    legend_elements = [
        mlines.Line2D([], [], color='black', marker='o', markersize=4, linestyle='None', label='True Data'),
        mlines.Line2D([], [], color='gray', linestyle=':', linewidth=2, label='Split')
    ]
    
    order_list = [
        "ARIMA", "SEIR", "DLinear (real)", "DLinear (aug)", "DLinear (comb)",
        "LSTM (real)", "LSTM (aug)", "LSTM (comb)",
        "Autoformer (real)", "Autoformer (aug)", "Autoformer (comb)",
        "TabPFN-TS", "Ensemble", "RespiCast"
    ]
    for lbl in order_list:
        if lbl in plotted_global_labels:
            legend_elements.append(mlines.Line2D([], [], color=plotted_global_labels[lbl], linewidth=2.5, label=lbl))

    fig.subplots_adjust(wspace=0.2, hspace=0.35, bottom=0.15)
    fig.legend(handles=legend_elements, loc="lower center", ncol=min(7, len(legend_elements)), bbox_to_anchor=(0.5, 0.02), fontsize=11, frameon=False)
    
    metric_display = "IS" if metric_type == "WIS80_mean" else metric_type
    fig.suptitle(f"Forecast Horizon {horizon}: RespiCast vs. Best Alternative (By {metric_display})", fontsize=15, fontweight="bold", y=0.93)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=800, bbox_inches="tight")
    plt.close(fig)
    print(f"\n Saved to: {out_path}")

if __name__ == "__main__":
    # 1. Based on MAE ranking
    plot_point_forecast_with_interval(
        metrics_csv="./results/metrics_tables/point_metrics_long_real_sim.csv",
        horizon=4,
        metric_type="MAE",
        out_path="./test_results/montages/all_countries_forecast_step4_MAE_Interval.pdf"
    )

    # 2. Based on IS ranking
    plot_point_forecast_with_interval(
        metrics_csv="./results/metrics_tables/point_metrics_long_real_sim.csv",
        horizon=4,
        metric_type="WIS80_mean",  # Note: This metric is called WIS80_mean in the CSV
        out_path="./test_results/montages/all_countries_forecast_step4_WIS_Interval.pdf"
    )
