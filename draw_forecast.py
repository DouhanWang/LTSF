import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

# -----------------------------
# 1. 核心配置
# -----------------------------
COUNTRIES = ["Belgium", "Czechia", "Denmark", "France", "Ireland", "Italy", "Netherlands", "Poland", "Romania"]

GRID_LAYOUT = [
    ["ARIMA_real",      "TabPFN_ts_real",       "SEIR_real"],
    ["DLinear_real",    "DLinear_augmented",    "DLinear_combined"],
    ["LSTM_real",       "LSTM_augmented",       "LSTM_combined"],
    ["Autoformer_real", "Autoformer_augmented", "Autoformer_combined"],
    ["Ensemble_real",   "Respicast_real",       None] 
]

# 严格按照你要求的预测长度
TARGET_LENGTHS = {1: 21, 2: 20, 3: 19, 4: 18} 

COUNTRY_YLIM = {
    "Belgium": (0, 2000), "Czechia": (0, 450), "Denmark": (0, 500),
    "France": (0, 1200), "Ireland": (0, 200), "Italy": (0, 3000),
    "Netherlands": (0, 300), "Poland": (0, 1800), "Romania": (0, 100),
}

# -----------------------------
# 配色方案 (统一更新版)
# -----------------------------
PAPER_COLORS = {
    # 基础基准：从中灰变为深灰，增加分量感
    "ARIMA_real": "#636363",      # Deep Gray
    "SEIR_real": "#008B8B",       # Dark Cyan (深青色，高冷严谨)

    # DLinear 系列：从浅蓝转向深海蓝/宝石蓝
    "DLinear_real": "#6BAED6",        # Medium Blue
    "DLinear_augmented": "#3182BD",   # Vibrant Blue
    "DLinear_combined": "#08519C",    # Deep Navy

    # LSTM 系列：从浅绿转向翡翠绿/森林绿
    "LSTM_real": "#74C476",           # Leaf Green
    "LSTM_augmented": "#31A354",      # Vibrant Green
    "LSTM_combined": "#006D2C",       # Deep Forest Green

    # Autoformer 系列：从淡橙转向琥珀色/铁锈橙
    "Autoformer_real": "#FD8D3C",       # Vibrant Orange
    "Autoformer_augmented": "#E6550D",  # Rich Burnt Orange
    "Autoformer_combined": "#A63603",   # Deep Rust

    # 特殊模型：提高色彩的明度对比
    "TabPFN_ts_real": "#D81B60",  # 深洋红 (从粉色升级，极具辨识度)
    "Respicast_real": "#B8860B",  # 暗金色
    "Ensemble_real": "#7570B3",   # 皇家紫 (突出综合模型的权威感)
    
    # 兜底颜色与真实值
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

# # -----------------------------
# # 2. 绘制大图 (完整更新版：5x3布局 + 隐藏空格子)
# # -----------------------------
# def plot_montage_for_country_step(results_dir: Path, out_dir: Path, country: str, step: int):
#     set_paper_style()
#     # 行数改为 5，高度从 10 增加到 12.5
#     fig, axes = plt.subplots(5, 3, figsize=(14, 12.5)) 
#     fig.subplots_adjust(bottom=0.15, hspace=0.15, wspace=0.1)
    
#     global_legend_dict = {}
    
#     # ========================================================
#     # 【核心1】提取雷打不动的 25 点 Master Ground Truth 底座
#     # (这部分就是你刚才报错缺失的代码)
#     # ========================================================
#     master_gt_series = fetch_ground_truth_series(results_dir, country, step)
#     if master_gt_series is not None:
#         master_gt_series = master_gt_series.tail(25)  # 严格锁定 25 个点
#         master_dates = master_gt_series.index.to_numpy()
#         master_y = master_gt_series.values
#     else:
#         master_dates = None
#         master_y = None

#     # 从字典中拿到当前 Step 对应的预测长度 (21/20/19/18)
#     pred_len = TARGET_LENGTHS.get(step, 18)

#     # 循环行数改为 5
#     for r in range(5):
#         for c in range(3):
#             ax = axes[r, c]
#             prefix = GRID_LAYOUT[r][c]
            
#             # 【新增】处理空的格子 (None)，遇到 None 直接隐藏并跳过
#             if prefix is None:
#                 ax.set_visible(False)
#                 continue

#             color, label = get_model_style(prefix)
            
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
            
#             ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#             ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 

#             if c > 0: ax.tick_params(labelleft=False)
#             else: ax.set_ylabel("Incidence")
                
#             # 【修改】X轴标签显示逻辑：如果是最后一行，或者它的正下方是空的格子，才显示X轴
#             is_bottom = (r == 4) or (r == 3 and GRID_LAYOUT[r+1][c] is None)
#             if not is_bottom: 
#                 ax.tick_params(labelbottom=False)
#             else: 
#                 plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

#             if country in COUNTRY_YLIM:
#                 ax.set_ylim(COUNTRY_YLIM[country])

#             # ========================================================
#             # 【核心2】在每个格子里，先把 25 个黑点和 分割线 画好
#             # ========================================================
#             if master_dates is not None:
#                 line_true, = ax.plot(master_dates, master_y, color='black', marker='o', markersize=3.5, linestyle='None', zorder=50)
#                 if "GroundTruth" not in global_legend_dict:
#                     global_legend_dict["GroundTruth"] = line_true
                
#                 if len(master_dates) > 3:
#                     # 将分割线死死钉在日期轴的 第4个点 (index 3) 上
#                     ax.axvline(x=master_dates[3], color='gray', linestyle=':', linewidth=1.5, alpha=0.8, zorder=49)

#             # ========================================================
#             # 【核心3】加载各模型的预测值，严格按字典长度截断
#             # ========================================================
#             folder_path = results_dir / f"{prefix}_{country}_ILI"
#             csv_path = folder_path / f"rolling_pred_step{step}.csv"
#             if not csv_path.exists(): csv_path = folder_path / f"rolling_pred_step_{step}.csv"
            
#             if not csv_path.exists():
#                 ax.set_title(f"{label} (Missing)", fontsize=11, color="gray")
#                 continue
                
#             df = pd.read_csv(csv_path)
#             date_col = next((col for col in ["date", "time", "timestamp"] if col in df.columns), None)
#             if not date_col: continue
            
#             df["_date"] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True)
            
#             pred_col = next((col for col in ["pred", "0.5", "target", "median"] if col in df.columns), None)
#             if not pred_col: continue
            
#             # 清除空值后，精准截取尾部 pred_len (21/20/19/18) 个点！
#             df_valid = df.dropna(subset=[pred_col])
#             df_pred = df_valid.tail(pred_len).copy()
            
#             pred_dates = df_pred["_date"].to_numpy()
#             y_pred = df_pred[pred_col].values
            
#             l_col = next((col for col in df_pred.columns if "lower" in col.lower() or "0.1" in col), None)
#             u_col = next((col for col in df_pred.columns if "upper" in col.lower() or "0.9" in col), None)
#             y_low = df_pred[l_col].values if l_col else None
#             y_up = df_pred[u_col].values if u_col else None

#             # 将纯净长度的预测线叠加上去
#             line_pred, = ax.plot(pred_dates, y_pred, color=color, linewidth=1.5, zorder=20)
#             if label not in global_legend_dict:
#                 global_legend_dict[label] = line_pred
                
#             if y_low is not None and y_up is not None:
#                 ax.fill_between(pred_dates, y_low, y_up, color=color, alpha=0.2, zorder=10)
                
#             ax.text(0.05, 0.9, label, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')

#     fig.legend(
#         list(global_legend_dict.values()), list(global_legend_dict.keys()),
#         loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.02),
#         frameon=False, fontsize=11
#     )
    
#     fig.suptitle(f"Forecast for {country} (Step {step})", y=0.96, fontsize=16, fontweight='bold')
#     save_path = out_dir / f"{country}_step{step}_montage.png"
#     plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     plt.close()
#     print(f"[OK] Saved {save_path}")
def plot_montage_for_country_step(results_dir: Path, out_dir: Path, country: str, step: int):
    set_paper_style()
    
    # 创建画布
    fig = plt.figure(figsize=(14, 15)) 
    
    # 5行6列网格，只有最底部一行高度增加，留足空间给斜向日期
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

    # 遍历 5 行
    for r in range(5):
        row_content = GRID_LAYOUT[r]
        valid_models = [m for m in row_content if m is not None]
        
        for i, prefix in enumerate(valid_models):
            # --- 核心布局逻辑：处理居中 ---
            if r < 4:
                ax = fig.add_subplot(gs[r, i*2 : (i+1)*2])
            else:
                ax = fig.add_subplot(gs[r, i*2 + 1 : (i+1)*2 + 1])

            color, label = get_model_style(prefix)
            
            # --- 坐标轴基础设置 ---
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            # Y轴标签逻辑：每行第一个图显示
            if i > 0: ax.tick_params(labelleft=False)
            else: ax.set_ylabel("Incidence")

            # ==========================================
            # 【修改1】：前四行全部隐藏 X 轴日期
            # ==========================================
            if r < 4:
                ax.tick_params(labelbottom=False)

            if country in COUNTRY_YLIM:
                ax.set_ylim(COUNTRY_YLIM[country])

            # 模型名称放在 Y 轴正上方
            ax.set_title(label, loc='left', fontsize=12, fontweight='bold', pad=10)

            # --- 绘制真实值和分割线 ---
            if master_dates is not None:
                line_true, = ax.plot(master_dates, master_y, color='black', marker='o', 
                                    markersize=3.5, linestyle='None', zorder=50)
                if "GroundTruth" not in global_legend_dict:
                    global_legend_dict["GroundTruth"] = line_true
                if len(master_dates) > 3:
                    ax.axvline(x=master_dates[3], color='gray', linestyle=':', linewidth=1.5, alpha=0.8, zorder=49)

            # --- 加载数据并绘制预测线 ---
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

            # ==========================================
            # 【修改2】：等所有线都画完了，再对最后一行的 X 轴进行倾斜
            # 这样就不会有漏网之鱼（比如 2024-12）横着了！
            # ==========================================
            if r == 4:
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # 图例设置
    fig.legend(list(global_legend_dict.values()), list(global_legend_dict.keys()),
               loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=11)
    
    # 国家名称主标题
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