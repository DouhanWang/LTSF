import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

# ==========================================
# 1. 颜色与基础配置
# ==========================================
PAPER_COLORS = {
    "ARIMA_real":       "#636363",
    "DLinear_real":     "#6BAED6",
    "DLinear_aug":      "#3182BD",
    "DLinear_comb":     "#08519C",
    "LSTM_real":        "#74C476",
    "LSTM_aug":         "#31A354",
    "LSTM_comb":        "#006D2C",
    "Autoformer_real":  "#FD8D3C",
    "Autoformer_aug":   "#E6550D",
    "Autoformer_comb":  "#A63603",
    "TabPFN_ts_real":   "#D81B60",
    "Respicast_real":   "#B8860B",
    "ensemble_real":    "#7570B3",
}

DEFAULT_COUNTRIES = [
    "Belgium", "Czechia", "Denmark", "France", "Ireland",
    "Italy", "Netherlands", "Poland", "Romania",
]

COUNTRY_YLIM = {
    "Belgium":     (0, 2000), "Czechia":     (0, 500),  "Denmark":     (0, 500),
    "France":      (0, 1200), "Ireland":     (0, 200),  "Italy":       (0, 3000),
    "Netherlands": (0, 250),  "Poland":      (0, 2000), "Romania":     (0, 100),
}

# 用精确 key 匹配，避免子字符串误匹配 (0.9 ⊂ 0.975 等问题)
LOWER_KEYS = {"lower", "lower_80", "lower_90", "q0.1", "0.1", "q_0.1", "lo_80", "lo_90", "pi_lower"}
UPPER_KEYS = {"upper", "upper_80", "upper_90", "q0.9", "0.9", "q_0.9", "hi_80", "hi_90", "pi_upper"}


def _paper_style_rcparams():
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "axes.linewidth": 1.0,
    })


def get_model_identity(method, setting):
    m = str(method).strip().lower()
    s = str(setting).strip().lower() if pd.notna(setting) else ""

    if m == "tabpfn_ts": return "TabPFN-TS", "TabPFN_ts_real"
    if m == "ensemble":  return "Ensemble",  "ensemble_real"
    if m == "respicast": return "RespiCast", "Respicast_real"
    if m == "naive":     return "Naive",     "ARIMA_real"
    if m == "arima":     return "ARIMA",     "ARIMA_real"

    if m == "dlinear":     m_display = "DLinear"
    elif m == "lstm":      m_display = "LSTM"
    elif m == "autoformer": m_display = "Autoformer"
    else:                  m_display = str(method)

    if s == "augmented":   s_display, s_tag = "aug",  "aug"
    elif s == "combined":  s_display, s_tag = "comb", "comb"
    else:                  s_display, s_tag = "real", "real"

    return f"{m_display} ({s_display})", f"{m_display}_{s_tag}"


# ==========================================
# 2. 与 all_metrics.py 对齐的数据读取器
# ==========================================
def load_aligned_from_csv(csv_path):
    """
    复现 all_metrics.py 的 align_by_pred_start 逻辑：
    - pred 的第一个有效（finite）值为 start
    - y_true/y_pred 均从 start 开始取，长度取 min
    返回 dict，或 None（读取失败）
    """
    p = Path(str(csv_path).replace("\\", "/"))
    if not p.exists():
        return None

    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}

    pcol = next((cols[k] for k in ["pred", "y_pred", "forecast", "target", "0.5"] if k in cols), None)
    tcol = next((cols[k] for k in ["y_true", "true", "gt", "incidenza"]           if k in cols), None)
    dcol = next((cols[k] for k in ["date", "dates", "timestamp", "target_date"]   if k in cols), None)

    if not pcol:
        return None

    p_vals = pd.to_numeric(df[pcol], errors="coerce").values
    t_vals = pd.to_numeric(df[tcol], errors="coerce").values if tcol else None
    d_vals = df[dcol].astype(str).values if dcol else np.array([""] * len(df))

    # --- align_by_pred_start ---
    valid_p = np.where(np.isfinite(p_vals))[0]
    if len(valid_p) == 0:
        return None
    start = int(valid_p[0])

    p_aligned = p_vals[start:]
    # 剥尾部 NaN
    valid_end = np.where(np.isfinite(p_aligned))[0]
    if len(valid_end) == 0:
        return None
    n = int(valid_end[-1]) + 1
    p_aligned = p_aligned[:n]

    t_aligned = t_vals[start:start + n] if t_vals is not None else None

    # --- 区间列：精确 key 匹配，不做子字符串 ---
    lcol = next((cols[k] for k in LOWER_KEYS if k in cols), None)
    ucol = next((cols[k] for k in UPPER_KEYS if k in cols), None)
    l_aligned = pd.to_numeric(df[lcol], errors="coerce").values[start:start + n] if lcol else None
    u_aligned = pd.to_numeric(df[ucol], errors="coerce").values[start:start + n] if ucol else None

    return {
        "t_full":    t_vals,     # 完整的 y_true 数组（含前导 NaN），用于背景真实值
        "d_full":    d_vals,     # 完整的日期数组
        "t_aligned": t_aligned,  # y_true[start:start+n]
        "p_aligned": p_aligned,  # y_pred[start:start+n]
        "l_aligned": l_aligned,
        "u_aligned": u_aligned,
        "start":     start,      # pred 在完整数组中的起始 index
        "n":         n,
    }


# ==========================================
# 3. 绘图主逻辑
# ==========================================
def plot_point_forecast_with_interval(
        metrics_csv="./results/metrics_tables/point_metrics_long_real_sim.csv",
        horizon=4,
        out_path="./test_results/montages/all_countries_forecast_step4_MAE_Interval.png"
):
    _paper_style_rcparams()

    if not os.path.exists(metrics_csv):
        print(f"Error: 找不到总表 {metrics_csv}")
        return

    df_metrics = pd.read_csv(metrics_csv)
    df_metrics["train_setting"] = df_metrics["train_setting"].fillna("").astype(str)
    df_sub = df_metrics[
        (df_metrics["dataset_type"] == "real") &
        (df_metrics["step"] == horizon) &
        (df_metrics["metric"] == "MAE")
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
        # 1. 读取 Ground Truth（从 Naive 的 source_file）
        # ----------------------------------------------------
        naive_row = df_c[df_c["method"].str.lower() == "naive"]
        if naive_row.empty:
            ax.set_title(f"{country} (No Naive GT)", loc="left")
            continue

        naive_data = load_aligned_from_csv(naive_row.iloc[0]["source_file"])
        if naive_data is None or naive_data["t_full"] is None:
            ax.set_title(f"{country} (GT Read Failed)", loc="left")
            continue

        t_full = naive_data["t_full"]
        d_full = naive_data["d_full"]

        # 去掉尾部 NaN，确定真实值的有效长度
        valid_t = np.where(np.isfinite(t_full))[0]
        if len(valid_t) == 0:
            ax.set_title(f"{country} (GT All NaN)", loc="left")
            continue
        total_len = int(valid_t[-1]) + 1
        t_display = t_full[:total_len]
        d_display = d_full[:total_len]

        x_true = np.arange(total_len)
        ax.plot(x_true, t_display, color="black", marker="o", markersize=3.5,
                alpha=0.9, linestyle="None", zorder=50)

        # split 线：用 Naive 自身 pred 的 start（即预测期起点）
        naive_start = naive_data["start"]
        if naive_start < total_len:
            ax.axvline(x=naive_start, color="gray", linestyle=":", linewidth=1.5,
                       alpha=0.8, zorder=49)

        # ----------------------------------------------------
        # 2. 查表：RespiCast + MAE 最优其他模型
        # ----------------------------------------------------
        respi_row  = df_c[df_c["method"].str.lower() == "respicast"]
        others     = df_c[df_c["method"].str.lower() != "respicast"]

        rows_to_plot = []
        if not respi_row.empty:
            rows_to_plot.append(respi_row.iloc[0])
        if not others.empty:
            best_row = others.loc[others["value"].idxmin()]
            rows_to_plot.append(best_row)
            ts = best_row["train_setting"]
            print(f"[{country}] MAE 最强: {best_row['method']} {ts} "
                  f"(MAE: {best_row['value']:.2f})")

        # ----------------------------------------------------
        # 3. 用 align_by_pred_start 画每个模型的预测
        # ----------------------------------------------------
        for row in rows_to_plot:
            data = load_aligned_from_csv(row["source_file"])
            if data is None:
                continue

            start = data["start"]
            p     = data["p_aligned"]
            n     = data["n"]
            l     = data["l_aligned"]
            u     = data["u_aligned"]

            # 对齐到 t_full 的坐标系（裁掉超出真实值范围的部分）
            end_in_full = start + n
            if end_in_full > total_len:
                clip = total_len - start
                if clip <= 0:
                    continue
                p = p[:clip]
                if l is not None: l = l[:clip]
                if u is not None: u = u[:clip]

            x_pred = np.arange(start, start + len(p))

            label, color_key = get_model_identity(row["method"], row["train_setting"])
            color = PAPER_COLORS.get(color_key, "#333333")

            ax.plot(x_pred, p, color=color, linewidth=2.0, linestyle="-",
                    alpha=0.9, zorder=20)
            if l is not None and u is not None:
                ax.fill_between(x_pred, l, u, color=color, alpha=0.2, zorder=10)

            plotted_global_labels[label] = color

        # ----------------------------------------------------
        # 4. X 轴：用真实日期
        # ----------------------------------------------------
        tick_idx = np.linspace(0, total_len - 1, min(6, total_len), dtype=int)
        ax.set_xticks(tick_idx)
        formatted_dates = []
        for i in tick_idx:
            raw = str(d_display[i])
            try:
                dt = pd.to_datetime(raw, format="mixed", dayfirst=True)
                formatted_dates.append(dt.strftime("%Y-%m"))
            except Exception:
                formatted_dates.append(raw[:7])
        ax.set_xticklabels(formatted_dates, rotation=25, ha="right")

        ax.set_xlim(-0.5, total_len - 0.5)
        ax.set_title(country, loc="left", fontsize=13, fontweight="bold")
        if country in COUNTRY_YLIM:
            ax.set_ylim(COUNTRY_YLIM[country])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        r, c = divmod(idx, 3)
        if c == 0: ax.set_ylabel("Incidence", fontsize=11)
        if r == 2: ax.set_xlabel("Date", fontsize=11)

    # ----------------------------------------------------
    # 5. 全局图例
    # ----------------------------------------------------
    legend_elements = [
        mlines.Line2D([], [], color="black", marker="o", markersize=4,
                      linestyle="None", label="True Data"),
        mlines.Line2D([], [], color="gray", linestyle=":", linewidth=1.5, label="Split"),
    ]
    order_list = [
        "ARIMA", "DLinear (real)", "DLinear (aug)", "DLinear (comb)",
        "LSTM (real)", "LSTM (aug)", "LSTM (comb)",
        "Autoformer (real)", "Autoformer (aug)", "Autoformer (comb)",
        "TabPFN-TS", "Ensemble", "RespiCast",
    ]
    for lbl in order_list:
        if lbl in plotted_global_labels:
            legend_elements.append(
                mlines.Line2D([], [], color=plotted_global_labels[lbl],
                              linewidth=2.5, label=lbl)
            )

    fig.subplots_adjust(wspace=0.2, hspace=0.35, bottom=0.15)
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=min(7, len(legend_elements)),
               bbox_to_anchor=(0.5, 0.02), fontsize=11, frameon=False)
    fig.suptitle(
        f"Forecast Horizon {horizon}: RespiCast vs. Best Alternative (By MAE Table)",
        fontsize=15, fontweight="bold", y=0.93,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[完成] 图表已保存至: {out_path}")


if __name__ == "__main__":
    plot_point_forecast_with_interval(
        metrics_csv="./results/metrics_tables/point_metrics_long_real_sim.csv",
        horizon=4,
        out_path="./test_results/montages/all_countries_forecast_step4_MAE_Interval.png"
    )
