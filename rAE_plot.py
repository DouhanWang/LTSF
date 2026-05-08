import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ==========================================
# 优雅的低饱和学术配色字典 (用于云雨图)
# ==========================================
PAPER_COLORS = {
    # 基础基准：从中灰变为深灰，增加分量感
    "ARIMA_real": "#636363",  # Deep Gray
    "SEIR_real": "#008B8B",   # Dark Cyan

    # DLinear 系列：从浅蓝转向深海蓝/宝石蓝
    "DLinear_real": "#6BAED6",  # Medium Blue
    "DLinear_aug": "#3182BD",   # Vibrant Blue
    "DLinear_comb": "#08519C",  # Deep Navy

    # LSTM 系列：从浅绿转向翡翠绿/森林绿
    "LSTM_real": "#74C476",     # Leaf Green
    "LSTM_aug": "#31A354",      # Vibrant Green
    "LSTM_comb": "#006D2C",     # Deep Forest Green

    # Autoformer 系列：从淡橙转向琥珀色/铁锈橙
    "Autoformer_real": "#FD8D3C", # Vibrant Orange
    "Autoformer_aug": "#E6550D",  # Rich Burnt Orange
    "Autoformer_comb": "#A63603", # Deep Rust

    # 特殊模型：提高色彩的明度对比
    "TabPFN_ts_real": "#D81B60",  # 深洋红 (从粉色升级，极具辨识度)
    "Respicast_real": "#B8860B",
    "Ensemble_real": "#7570B3",   # 皇家紫 (从浅紫升级，突出权威感)
}

DEFAULT_COUNTRIES = [
    "Belgium", "Czechia", "Denmark", "France", "Ireland",
    "Italy", "Netherlands", "Poland", "Romania",
]

DEFAULT_METHODS = [
    ("ARIMA_real", "ARIMA"),
    ("SEIR_real", "SEIR"),
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
    ("Ensemble_real", "Ensemble"),  # 修正：小写 e
    ("Respicast_real", "RespiCast"),  # 修正：大写 R
]

def get_paper_color(tag):
    return PAPER_COLORS.get(tag, "#CCCCCC")


def _list_run_dirs(results_root: str, settings_contains=None):
    run_dirs = [d for d in glob.glob(os.path.join(results_root, "*")) if os.path.isdir(d)]
    if settings_contains:
        if isinstance(settings_contains, str):
            run_dirs = [d for d in run_dirs if settings_contains in os.path.basename(d)]
        else:
            run_dirs = [
                d for d in run_dirs
                if all(tok in os.path.basename(d) for tok in settings_contains)
            ]
    return run_dirs


def _find_latest_run(run_dirs, country: str, method_tag: str):
    c0 = country.lower()
    m0 = method_tag.lower()
    cand = []
    for d in run_dirs:
        b = os.path.basename(d).lower()
        if (c0 in b) and (m0 in b):
            cand.append(d)
    if not cand:
        return None
    return sorted(cand, key=lambda x: os.path.getmtime(x))[-1]

def _coerce_dates(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        out = []
        for v in list(x):
            if isinstance(v, np.ndarray) and v.shape == ():
                v = v.item()
            out.append(str(v))
        return out
    return None


# ==========================================
# 从 CSV 中分别读取 pred 和 true
# ==========================================
def _load_data_from_csv(run_dir: str, method_tag: str, horizon: int):

    filename = f"rolling_pred_step{int(horizon)}.csv"

    p = os.path.join(run_dir, filename)
    if not os.path.exists(p):
        return None, None, None

    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[warn] Failed to read {p}: {e}")
        return None, None, None

    # 找 Date
    date_cols = ["date", "dates", "timestamp", "time", "ds", "target_date", "forecast_date"]
    date_col = next((c for c in date_cols if c in df.columns), None)

    # 找 True
    true_cols = ["true", "target", "actual", "y", "y_true"]
    true_col = next((c for c in true_cols if c in df.columns), None)

    # 找 Pred
    pred_cols = ["pred", "forecast", "mean", "0.5", "50", "yhat", "y_pred", "median"]
    pred_col = next((c for c in pred_cols if c in df.columns), None)

    if not pred_col:
        print(f"[warn] 缺失预测值列 in {p}. 当前列: {df.columns.tolist()}")
        return None, None, None

    dates = _coerce_dates(df[date_col].values) if date_col else None
    pred_vals = pd.to_numeric(df[pred_col], errors='coerce').values
    true_vals = pd.to_numeric(df[true_col], errors='coerce').values if true_col else None

    return pred_vals, true_vals, dates


def _align_by_dates(val_a, dates_a, val_b, dates_b):
    if val_a is None or val_b is None:
        return None, None

    # 如果任一缺乏日期，按照“倒数”对齐（时序序列常在末端对齐）
    if dates_a is None or dates_b is None:
        n = min(len(val_a), len(val_b))
        # 🔴 修改点：改成 [-n:] 从后往前取
        return np.asarray(val_a[-n:], float), np.asarray(val_b[-n:], float)

    map_a = {str(d): i for i, d in enumerate(dates_a)}
    map_b = {str(d): i for i, d in enumerate(dates_b)}

    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    if len(common) == 0:
        return None, None

    ia = [map_a[k] for k in common]
    ib = [map_b[k] for k in common]

    return np.asarray(val_a, float)[ia], np.asarray(val_b, float)[ib]


def _paper_style_rcparams():
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ==========================================
# 绘制 Relative Absolute Error 的云雨图
# ==========================================
def plot_relative_ae_grid_3x3(
        results_root="./results",
        countries=None,
        horizon=1,
        methods=None,
        baseline_tag="Naive_real",
        settings_contains=None,
        out_path="./test_results/montages/relae_grid_step1.png",
        dpi=800,
):
    _paper_style_rcparams()

    if countries is None:
        countries = DEFAULT_COUNTRIES
    if methods is None:
        methods = DEFAULT_METHODS

    run_dirs = _list_run_dirs(results_root, settings_contains=settings_contains)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    axes = axes.ravel()

    FIXED_ORDER = methods[::-1]

    for idx, country in enumerate(countries):
        ax = axes[idx]

        # ---- baseline ----
        base_dir = _find_latest_run(run_dirs, country, baseline_tag)
        if base_dir is None:
            ax.set_axis_off()
            continue

        base_pred, base_true, base_dates = _load_data_from_csv(base_dir, baseline_tag, horizon)
        if base_pred is None or base_true is None:
            ax.set_axis_off()
            continue

        base_ae = np.abs(base_pred - base_true)
        base_ae = np.where(base_ae == 0, 1e-9, base_ae)  # 防止除零

        rel_data = []
        labels = []
        tags_used = []

        for method_tag, method_label in FIXED_ORDER:
            if method_tag == baseline_tag:
                continue

            m_dir = _find_latest_run(run_dirs, country, method_tag)
            if m_dir is None:
                continue

            m_pred, m_true, m_dates = _load_data_from_csv(m_dir, method_tag, horizon)
            if m_pred is None:
                continue

# 🔴 核心逻辑：如果模型缺少 True 值，借用 Naive 的 True 值（根据绝对日期匹配！）
            if m_true is None:
                if base_dates is not None and m_dates is not None:
                    # 1. 建立 Naive 的 {日期字符串: 真实值} 字典
                    base_true_dict = {str(d): val for d, val in zip(base_dates, base_true)}
                    
                    # 2. 严格拿着当前模型预测的日期去字典里取值
                    m_true = np.array([base_true_dict.get(str(d), np.nan) for d in m_dates])
                else:
                    print(f"[warn] {country} {method_tag} 缺少 True 值且无法进行日期匹配，跳过。")
                    continue

            m_ae = np.abs(m_pred - m_true)

            # 对齐当前模型 AE 和 Base AE
            m_ae_aligned, base_ae_aligned = _align_by_dates(m_ae, m_dates, base_ae, base_dates)
            if m_ae_aligned is None or len(m_ae_aligned) == 0:
                continue

            base_ae_aligned = np.where(base_ae_aligned == 0, 1e-9, base_ae_aligned)
            rel = m_ae_aligned / base_ae_aligned
            rel = rel[np.isfinite(rel) & (rel > 0.01)]

            if rel.size == 0:
                continue

            rel_data.append(rel)
            labels.append(method_label)
            tags_used.append(method_tag)

        if not rel_data:
            ax.set_axis_off()
            continue

        ys = np.arange(1, len(rel_data) + 1)

        # # 1. 云 (Cloud)
        # vp = ax.violinplot(
        #     rel_data, positions=ys, vert=False, widths=0.8,
        #     bw_method=0.25, showmeans=False, showmedians=False, showextrema=False
        # )
        # offset_cloud = 0.05
        # for i, body in enumerate(vp["bodies"]):
        #     m = ys[i]
        #     path = body.get_paths()[0]
        #     path.vertices[:, 1] = np.clip(path.vertices[:, 1], m + offset_cloud, np.inf)
        #     color = get_paper_color(tags_used[i])
        #     body.set_facecolor(color)
        #     body.set_alpha(0.45)
        #     body.set_edgecolor(color)
        #     body.set_linewidth(1.0)

        # 2. 核与雨 (Box & Rain)
        for y_pos, d in zip(ys, rel_data):
            color = get_paper_color(tags_used[y_pos - 1])
            y_jitter = y_pos - 0.25 - np.random.uniform(0, 0.25, size=len(d))
            ax.scatter(d, y_jitter, s=6, alpha=0.35, color=color, edgecolors='none', zorder=1)

            q25, q50, q75 = np.quantile(d, [0.25, 0.5, 0.75])
            box_y = y_pos - 0.08
            # ax.plot([q25, q75], [box_y, box_y], color="#888888", linewidth=1.5, solid_capstyle="round", zorder=3)
            # ax.scatter(q50, box_y, color="white", s=18, zorder=4, edgecolor="#888888", linewidth=1.0)
            ax.plot([q25, q75], [box_y, box_y], color=color, linewidth=1.2, solid_capstyle="round", zorder=3)

            ax.scatter(
                q50, box_y,
                facecolors="none",  # 空心
                edgecolors=color,  # 边框用对应模型颜色
                s=18,
                zorder=4,
                linewidth=1.2
            )
        # 3. 坐标轴与背景
        ax.set_xscale('log', base=2)
        x_ticks = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
        ax.set_xticks(x_ticks)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlim(0.1, 4.0)

        ax.axvline(1.0, linestyle="--", linewidth=1.2, color="#888888", alpha=0.8, zorder=0)
        ax.grid(axis="x", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#DDDDDD")

        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=13)
        ax.set_title(country, loc="left", fontsize=14, fontweight="bold", pad=10)

        r, c = divmod(idx, 3)
        if c != 0:
            ax.tick_params(axis='y', left=False, labelleft=False)

        if r == 2:
            ax.set_xlabel("Relative Absolute Error (vs. Naive Real)", fontsize=14, color="#333333", labelpad=8)
        else:
            ax.tick_params(axis='x', bottom=True, labelbottom=False)

    fig.subplots_adjust(wspace=0.15, hspace=0.25)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved Relative AE grid: {out_path}")


if __name__ == "__main__":
    plot_relative_ae_grid_3x3(
        results_root="./results",
        countries=DEFAULT_COUNTRIES,
        horizon=1,
        methods=DEFAULT_METHODS,
        baseline_tag="Naive_real",
        settings_contains=None,
        out_path="./test_results/montages/relae_grid_step1_new.pdf",
        dpi=800,
    )