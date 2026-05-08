# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# -----------------------------
# 1. Settings
# -----------------------------
COUNTRIES = ["Belgium", "Czechia", "Denmark", "France", "Ireland", "Italy", "Netherlands", "Poland", "Romania"]

GRID_LAYOUT = [
    ["ARIMA_real",      "TabPFN_ts_real",       "SEIR_real"],
    ["DLinear_real",    "DLinear_augmented",    "DLinear_combined"],
    ["LSTM_real",       "LSTM_augmented",       "LSTM_combined"],
    ["Autoformer_real", "Autoformer_augmented", "Autoformer_combined"],
    ["Ensemble_real",   "Respicast_real",       None] 
]

# Prediction length
TARGET_LENGTHS = {1: 21, 2: 20, 3: 19, 4: 18} 

COUNTRY_YLIM = {
    "Belgium": (0, 2000), "Czechia": (0, 450), "Denmark": (0, 500),
    "France": (0, 1200), "Ireland": (0, 200), "Italy": (0, 3000),
    "Netherlands": (0, 300), "Poland": (0, 1800), "Romania": (0, 100),
}

# -----------------------------
# Color settings
# -----------------------------
PAPER_COLORS = {
    "ARIMA_real": "#636363",      # Deep Gray
    "SEIR_real": "#008B8B",       # Dark Cyan

    # DLinear
    "DLinear_real": "#6BAED6",        # Medium Blue
    "DLinear_augmented": "#3182BD",   # Vibrant Blue
    "DLinear_combined": "#08519C",    # Deep Navy

    # LSTM
    "LSTM_real": "#74C476",           # Leaf Green
    "LSTM_augmented": "#31A354",      # Vibrant Green
    "LSTM_combined": "#006D2C",       # Deep Forest Green

    # Autoformer
    "Autoformer_real": "#FD8D3C",       # Vibrant Orange
    "Autoformer_augmented": "#E6550D",  # Rich Burnt Orange
    "Autoformer_combined": "#A63603",   # Deep Rust

    
    "TabPFN_ts_real": "#D81B60", 
    "Respicast_real": "#B8860B",  
    "Ensemble_real": "#7570B3",  
    

    "Naive_real": "#9E9E9E",
    "default": "#2B2B2B"
}

def set_paper_style():
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def get_model_style(prefix):
    if prefix is None: return None, None
    m_name = prefix.split('_')[0]
    display_names = {"TabPFN": "TabPFN-TS", "Respicast": "RespiCast", "SEIR": "SEIR", "Ensemble": "Ensemble"}
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

def plot_montage_for_country_step(results_dir: Path, out_dir: Path, country: str, step: int):
    set_paper_style()
    

    fig = plt.figure(figsize=(14, 15)) 
    
    # 5 rows 6 columns
    gs = gridspec.GridSpec(5, 6, figure=fig, 
                           height_ratios=[1, 1, 1, 1, 1.2], 
                           hspace=0.4, wspace=0.3)
    
    global_legend_dict = {}
    master_gt_series = fetch_ground_truth_series(results_dir, country, step)
    
    if master_gt_series is not None:
        master_gt_series = master_gt_series.tail(25)
        master_dates = master_gt_series.index.to_numpy()
        master_y = master_gt_series.values
    else:
        master_dates = master_y = None

    pred_len = TARGET_LENGTHS.get(step, 18)


    for r in range(5):
        row_content = GRID_LAYOUT[r]
        valid_models = [m for m in row_content if m is not None]
        
        for i, prefix in enumerate(valid_models):

            if r < 4:
                ax = fig.add_subplot(gs[r, i*2 : (i+1)*2])
            else:
                ax = fig.add_subplot(gs[r, i*2 + 1 : (i+1)*2 + 1])

            color, label = get_model_style(prefix)
            
            # ---axis settings---
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            # Y-axis label only on the leftmost column
            if i > 0: ax.tick_params(labelleft=False)
            else: ax.set_ylabel("Incidence")


            if r < 4:
                ax.tick_params(labelbottom=False)

            if country in COUNTRY_YLIM:
                ax.set_ylim(COUNTRY_YLIM[country])


            ax.set_title(label, loc='left', fontsize=12, fontweight='bold', pad=10)

            # --- plot true values and split line ---
            if master_dates is not None:
                line_true, = ax.plot(master_dates, master_y, color='black', marker='o', 
                                    markersize=3.5, linestyle='None', zorder=50)
                if "GroundTruth" not in global_legend_dict:
                    global_legend_dict["GroundTruth"] = line_true
                if len(master_dates) > 3:
                    ax.axvline(x=master_dates[3], color='gray', linestyle=':', linewidth=1.5, alpha=0.8, zorder=49)

            # --- load data and plot prediction lines ---
            folder_path = results_dir / f"{prefix}_{country}_ILI"
            csv_path = folder_path / f"rolling_pred_step{step}.csv"
            if not csv_path.exists(): csv_path = folder_path / f"rolling_pred_step_{step}.csv"
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                pred_col = next((c for c in ["pred", "0.5", "target", "median"] if c in df.columns), None)
                date_col = next((c for c in ["date", "time", "timestamp"] if c in df.columns), None)
                
                if pred_col and date_col:
                    df["_date"] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True)
                    df_pred = df.dropna(subset=[pred_col]).tail(pred_len)
                    
                    line_pred, = ax.plot(df_pred["_date"], df_pred[pred_col], color=color, linewidth=1.5, zorder=20)
                    if label not in global_legend_dict:
                        global_legend_dict[label] = line_pred
                    
                    l_col = next((c for c in df_pred.columns if "lower" in c.lower() or "0.1" in c), None)
                    u_col = next((c for c in df_pred.columns if "upper" in c.lower() or "0.9" in c), None)
                    if l_col and u_col:
                        ax.fill_between(df_pred["_date"], df_pred[l_col], df_pred[u_col], color=color, alpha=0.2)
            else:
                ax.set_title(f"{label} (Missing)", loc='left', fontsize=12, fontweight='bold', color='gray', pad=10)


            if r == 4:
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')


    fig.legend(list(global_legend_dict.values()), list(global_legend_dict.keys()),
               loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=11)
    

    fig.suptitle(f"Forecast for {country} (Step {step})", y=0.96, fontsize=16, fontweight='bold')
    
    plt.savefig(out_dir / f"{country}_step{step}_montage.png", bbox_inches='tight', dpi=300)
    plt.close()

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