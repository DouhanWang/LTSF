import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# -----------------------------
# Config (edit here if needed)
# -----------------------------
DEFAULT_COUNTRIES = [
    "Belgium", "Czechia", "Denmark", "France", "Ireland",
    "Italy", "Netherlands", "Poland", "Romania",
]


DEFAULT_METHODS = [
    ("Naive_real", "Naive (Real)"),
    ("ARIMA_real", "ARIMA (Real)"),
    ("SEIR_real", "SEIR"),
    ("DLinear_real", "DLinear (Real)"),
    ("DLinear_aug", "DLinear (Aug)"),
    ("DLinear_comb", "DLinear (Comb)"),
    ("LSTM_real", "LSTM (Real)"),
    ("LSTM_aug", "LSTM (Aug)"),
    ("LSTM_comb", "LSTM (Comb)"),
    ("Autoformer_real", "Autoformer (Real)"),
    ("Autoformer_aug", "Autoformer (Aug)"),
    ("Autoformer_comb", "Autoformer (Comb)"),
    ("TabPFN_ts_real", "TabPFN-TS (Real)"),
    ("Respicast_real", "Respicast (Real)"),
    ("Ensemble_real", "Ensemble"),
]

# 给每个“方法族”分配一个 colormap（不同色系）
FAMILY_CMAP = {
    "ARIMA": "Purples",
    "DLinear": "Blues",
    "LSTM": "Greens",
    "Autoformer": "Oranges",
    "TabPFN_ts": "Reds",
    "Ensemble": "Greys",
    "respicast": "YlGnBu",
}

# real/aug/combined 的深浅档位（浅->深）
VAR_LEVEL = {
    "real": 0.35,
    "augmented": 0.55,
    "combined": 0.75,
}

def method_color(method_tag: str):
    """
    method_tag examples:
      DLinear_real, DLinear_augmented, DLinear_combined
      ARIMA_real, TabPFN_ts_real, Ensemble_real, respicast_real
    """
    parts = method_tag.split("_")
    # family 可能是 TabPFN_ts 这种带下划线的，稳妥处理：
    if method_tag.startswith("TabPFN_ts"):
        family = "TabPFN_ts"
        variant = parts[-1] if parts[-1] in VAR_LEVEL else "real"
    else:
        family = parts[0]
        variant = parts[-1] if parts[-1] in VAR_LEVEL else "real"

    cmap_name = FAMILY_CMAP.get(family, "Greys")
    cmap = mpl.colormaps[cmap_name]
    lvl = VAR_LEVEL.get(variant, 0.55)
    return cmap(lvl)
def _list_run_dirs(results_root: str, settings_contains=None):
    run_dirs = [d for d in glob.glob(os.path.join(results_root, "*")) if os.path.isdir(d)]
    if settings_contains:
        if isinstance(settings_contains, str):
            run_dirs = [d for d in run_dirs if settings_contains in os.path.basename(d)]
        else:
            # list/tuple of tokens: all must appear
            run_dirs = [
                d for d in run_dirs
                if all(tok in os.path.basename(d) for tok in settings_contains)
            ]
    return run_dirs


def _find_latest_run(run_dirs, country: str, method_tag: str):
    """
    Heuristic matching:
      - must contain country substring
      - must contain method_tag substring
    Choose latest by mtime.
    """
    cand = [
        d for d in run_dirs
        if (country in os.path.basename(d)) and (method_tag in os.path.basename(d))
    ]
    if not cand:
        return None
    return sorted(cand, key=lambda x: os.path.getmtime(x))[-1]


def _coerce_dates(x):
    """
    Convert various date containers to a list of python objects comparable by str().
    Accepts list of datetime/date, numpy datetime64, pandas Timestamp, etc.
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    if isinstance(x, (list, tuple, np.ndarray)):
        out = []
        for v in list(x):
            if isinstance(v, np.ndarray) and v.shape == ():
                v = v.item()
            out.append(v)
        return out
    return None


def _load_wis_from_run(run_dir: str, wis_file_prefix: str, horizon: int):
    p = os.path.join(run_dir, f"{wis_file_prefix}{int(horizon)}.npy")
    if not os.path.exists(p):
        return None, None

    obj = np.load(p, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()

    dates = None

    # if dict: extract wis + dates
    if isinstance(obj, dict):
        # --- dates keys ---
        for dk in ["dates", "date", "timestamp", "time", "ds", "target_dates", "forecast_dates"]:
            if dk in obj:
                dates = _coerce_dates(obj[dk])
                break

        # --- wis keys ---
        key_candidates = [
            "wis", "WIS", "wis80", "wis_80", "wis80_point", "wis_point",
            "scores", "score", "values", "arr", "array", "data"
        ]
        picked = None
        for k in key_candidates:
            if k in obj:
                picked = obj[k]
                break
        if picked is None:
            for v in obj.values():
                if isinstance(v, (list, tuple, np.ndarray)):
                    picked = v
                    break
        if picked is None:
            print(f"[warn] {p} is dict but no array-like WIS found. keys={list(obj.keys())}")
            return None, dates
        obj = picked
        if isinstance(obj, np.ndarray) and obj.shape == ():
            obj = obj.item()

    # normalize to float array
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        try:
            obj = np.array(obj, dtype=float)
        except Exception:
            flat = []
            for x in obj.ravel():
                if x is None:
                    continue
                if isinstance(x, (list, tuple, np.ndarray)):
                    flat.extend(np.asarray(x).ravel().tolist())
                else:
                    flat.append(x)
            obj = np.asarray(flat, dtype=float)

    if isinstance(obj, (list, tuple)):
        obj = np.asarray(obj, dtype=float)

    try:
        arr = np.asarray(obj, dtype=float).ravel()
    except Exception:
        print(f"[warn] cannot cast to float: {p} type={type(obj)}")
        return None, dates

    return arr, dates

def _align_by_dates(wis_a, dates_a, wis_b, dates_b):
    """
    Align two WIS arrays by intersection of dates.
    Returns (a_aligned, b_aligned). If dates missing, falls back to min-length truncation.
    """
    if wis_a is None or wis_b is None:
        return None, None

    # fallback if any dates missing
    if dates_a is None or dates_b is None:
        n = min(len(wis_a), len(wis_b))
        return np.asarray(wis_a[:n], float), np.asarray(wis_b[:n], float)

    # build index maps by stringified date (robust across date/datetime/Timestamp)
    map_a = {str(d): i for i, d in enumerate(dates_a)}
    map_b = {str(d): i for i, d in enumerate(dates_b)}

    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    if len(common) == 0:
        return None, None

    ia = [map_a[k] for k in common]
    ib = [map_b[k] for k in common]

    a_al = np.asarray(wis_a, float)[ia]
    b_al = np.asarray(wis_b, float)[ib]
    return a_al, b_al

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


def plot_relative_wis_paper_all(
    results_root="./results",
    countries=None,
    horizons=(1, 2, 3, 4),
    methods=None,
    baseline_tag="Naive_real",
    wis_file_prefix="wis80_point_step",
    settings_contains=None,   # set to None if you don't want filtering
    out_dir="./test_results/montages",
    dpi=300,
    ylim=None,               # e.g. (0.6, 1.6) or None for robust auto
):
    """
    Generate 9 countries x 4 steps = 36 plots:
      Relative IS = IS(method) / IS(baseline_tag)
    Using latest run folder for each (country, method_tag).
    Outliers disabled.
    """

    _paper_style_rcparams()

    if countries is None:
        countries = DEFAULT_COUNTRIES
    if methods is None:
        methods = DEFAULT_METHODS

    run_dirs = _list_run_dirs(results_root, settings_contains=settings_contains)

    # color palette (paper-friendly, consistent)
    # (only affects median line color per method; boxes are light gray)
    palette = [
        "#4E79A7", "#F28E2B", "#59A14F", "#E15759",
        "#B07AA1", "#76B7B2", "#EDC948", "#9C755F",
        "#BAB0AC", "#2E6F4E", "#8F63F4", "#D37295",
    ]
    method_colors = {methods[i][0]: palette[i % len(palette)] for i in range(len(methods))}

    for country in countries:
        for h in horizons:
            # --- baseline ---
            base_dir = _find_latest_run(run_dirs, country, baseline_tag)
            if base_dir is None:
                print(f"[skip] {country} step{h}: missing baseline run for {baseline_tag}")
                continue

            base_wis, base_dates = _load_wis_from_run(base_dir, wis_file_prefix, h)
            if base_wis is None:
                print(f"[skip] {country} step{h}: missing baseline file in {base_dir}")
                continue

            base_wis = np.where(base_wis == 0, 1e-9, base_wis)

            rel_data = []
            labels = []
            used_colors = []

            # --- methods ---
            for method_tag, method_label in methods:
                if method_tag == baseline_tag:
                    continue

                m_dir = _find_latest_run(run_dirs, country, method_tag)
                if m_dir is None:
                    print(f"[skip] {country} step{h}: missing run for {method_tag}")
                    continue

                wis, wis_dates = _load_wis_from_run(m_dir, wis_file_prefix, h)

                wis_aligned, base_aligned = _align_by_dates(wis, wis_dates, base_wis, base_dates)
                if wis_aligned is None:
                    print(f"[skip] {country} step{h}: no common dates for {method_tag}")
                    continue

                base_aligned = np.where(base_aligned == 0, 1e-9, base_aligned)
                rel = wis_aligned / base_aligned
                rel = rel[np.isfinite(rel)]
                rel = rel[np.isfinite(rel)]
                print(country, "step", h, method_tag, "aligned_n=", len(rel))
                if rel.size == 0:
                    print(f"[skip] {country} step{h}: empty rel for {method_tag}")
                    continue

                rel_data.append(rel)
                labels.append(method_label)
                used_colors.append(method_colors.get(method_tag, "#333333"))

            if not rel_data:
                print(f"[skip] {country} step{h}: no methods loaded")
                continue
            # -----------------------------
            # Sort methods by performance (lower median relWIS is better)
            # -----------------------------
            meds = np.array([np.median(d) for d in rel_data], dtype=float)
            order = np.argsort(meds)  # ascending: best -> worst

            rel_data = [rel_data[i] for i in order]
            labels = [labels[i] for i in order]

            # sequential greens: left light -> right dark
            k = len(rel_data)
            cmap = mpl.colormaps["Blues"]
            box_colors = [cmap(x) for x in np.linspace(0.30, 0.85, k)]  # light -> dark
            # -----------------------------
            # Dynamic y-limits: upper bound = max + padding
            # -----------------------------
            # -----------------------------
            # y-limits (median-based, robust)
            # -----------------------------
            # rel_data 已经是排序后的 list[np.ndarray]
            meds_sorted = np.array([np.median(d) for d in rel_data], dtype=float)

            y0 = 0.0  # 你也可以用 0.6
            stats = []
            for d in rel_data:
                q1, q2, q3 = np.quantile(d, [0.25, 0.5, 0.75])
                stats.append(q3)  # 用 Q3 当代表上沿
            ymax_med = float(np.max(stats))
            pad = max(0.20 * ymax_med, 0.30)
            y1 = max(ymax_med + pad, 1.25)

            # --- plot ---
            fig, ax = plt.subplots(figsize=(10.5, 3.9))

            bp = ax.boxplot(
                rel_data,
                tick_labels=labels,
                widths=0.55,
                showfliers=False,          # ✅ no outliers
                patch_artist=True,
                medianprops=dict(linewidth=2.2),
                whiskerprops=dict(linewidth=1.0, color="#2B2B2B"),
                capprops=dict(linewidth=1.1),
                boxprops=dict(linewidth=1.1),
            )
            for cap in bp["caps"]:
                cap.set_visible(False)
            # style: filled boxes with green gradient, consistent edges
            for i in range(len(rel_data)):
                bp["boxes"][i].set_facecolor(box_colors[i])
                bp["boxes"][i].set_edgecolor("#2B2B2B")
                bp["boxes"][i].set_alpha(0.95)

                bp["medians"][i].set_color("#1F1F1F")
                bp["medians"][i].set_linewidth(2.2)

            for w in bp["whiskers"]:
                w.set_color("#2B2B2B")
            # for c in bp["caps"]:
            #     c.set_color("#2B2B2B")

            # baseline reference
            ax.axhline(1.0, linestyle="--", linewidth=1.3, color="#666666", alpha=0.9, zorder=0)

            ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_ylabel("Relative IS (vs. Naive Real)")
            ax.set_ylim(y0, y1)
            ax.margins(y=0.02)  # 可选：再留一点点空白
            ax.set_title(f"{country}: Relative IS ({h}-step ahead)")

            plt.setp(ax.get_xticklabels(), rotation=18, ha="right")

            fig.tight_layout()

            # --- save ---
            save_dir = os.path.join(out_dir, country)
            os.makedirs(save_dir, exist_ok=True)

            png_path = os.path.join(save_dir, f"relwis_{country}_step{h}.png")
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)

            print(f"Saved: {png_path}")

# def plot_relative_wis_grid_3x3(
#     results_root="./results",
#     countries=None,          # 9个国家
#     horizon=1,               # 只画某一个step (1/2/3/4)
#     methods=None,
#     baseline_tag="Naive_real",
#     wis_file_prefix="wis80_point_step",
#     settings_contains=None,
#     out_path="./test_results/montages/relwis_grid_step1.png",
#     dpi=300,
#     ylim=None,               # (y0,y1) or None
# ):
#     _paper_style_rcparams()
#
#     if countries is None:
#         countries = DEFAULT_COUNTRIES
#     if methods is None:
#         methods = DEFAULT_METHODS
#     assert len(countries) == 9, "countries 必须正好9个，才能画 3x3"
#
#     run_dirs = _list_run_dirs(results_root, settings_contains=settings_contains)
#
#     fig, axes = plt.subplots(3, 3, figsize=(14, 8.5), sharey=False)
#     axes = axes.ravel()
#
#     # 先存一下所有country算出来的rel_data，用于统一ylim（可选）
#     all_ymax = []
#
#     for idx, country in enumerate(countries):
#         ax = axes[idx]
#
#         # ---- baseline ----
#         base_dir = _find_latest_run(run_dirs, country, baseline_tag)
#         if base_dir is None:
#             ax.set_axis_off()
#             continue
#         base_wis, base_dates = _load_wis_from_run(base_dir, wis_file_prefix, horizon)
#         if base_wis is None:
#             ax.set_axis_off()
#             continue
#         base_wis = np.where(base_wis == 0, 1e-9, base_wis)
#
#         rel_data = []
#         labels = []
#
#         for method_tag, method_label in methods:
#             if method_tag == baseline_tag:
#                 continue
#             m_dir = _find_latest_run(run_dirs, country, method_tag)
#             if m_dir is None:
#                 continue
#             wis, wis_dates = _load_wis_from_run(m_dir, wis_file_prefix, horizon)
#             wis_aligned, base_aligned = _align_by_dates(wis, wis_dates, base_wis, base_dates)
#             if wis_aligned is None:
#                 continue
#             base_aligned = np.where(base_aligned == 0, 1e-9, base_aligned)
#             rel = (wis_aligned / base_aligned)
#             rel = rel[np.isfinite(rel)]
#             if rel.size == 0:
#                 continue
#             rel_data.append(rel)
#             labels.append(method_label)
#
#         if not rel_data:
#             ax.set_axis_off()
#             continue
#
#         # ---- 排序：表现好（median小）放左边 ----
#         FIXED_ORDER = [
#             ("ARIMA_real", "ARIMA"),
#             ("DLinear_real", "DLinear (real)"),
#             ("DLinear_augmented", "DLinear (aug)"),
#             ("DLinear_combined", "DLinear (comb)"),
#             ("LSTM_real", "LSTM (real)"),
#             ("LSTM_augmented", "LSTM (aug)"),
#             ("LSTM_combined", "LSTM (comb)"),
#             ("Autoformer_real", "Autoformer (real)"),
#             ("Autoformer_augmented", "Autoformer (aug)"),
#             ("Autoformer_combined", "Autoformer (comb)"),
#             ("TabPFN_ts_real", "TabPFN-TS"),
#             ("Ensemble_real", "Ensemble"),
#             ("respicast_real", "RespiCast"),
#         ]
#         rel_data = []
#         labels = []
#         tags_used = []
#
#         for method_tag, method_label in FIXED_ORDER:
#             m_dir = _find_latest_run(run_dirs, country, method_tag)
#             if m_dir is None:
#                 continue
#             wis, wis_dates = _load_wis_from_run(m_dir, wis_file_prefix, horizon)
#             wis_aligned, base_aligned = _align_by_dates(wis, wis_dates, base_wis, base_dates)
#             if wis_aligned is None:
#                 continue
#
#             rel = (wis_aligned / np.where(base_aligned == 0, 1e-9, base_aligned))
#             rel = rel[np.isfinite(rel)]
#             if rel.size == 0:
#                 continue
#
#             rel_data.append(rel)
#             labels.append(method_label)
#             tags_used.append(method_tag)
#
#
#
#         # bp = ax.boxplot(
#         #     rel_data,
#         #     tick_labels=labels,
#         #     widths=0.55,
#         #     showfliers=False,
#         #     patch_artist=True,
#         #     medianprops=dict(linewidth=2.0, color="#1F1F1F"),
#         #     whiskerprops=dict(linewidth=1.0, color="#2B2B2B"),
#         #     capprops=dict(linewidth=1.0),
#         #     boxprops=dict(linewidth=1.0, color="#2B2B2B"),
#         # )
#         # for i, tag in enumerate(tags_used):
#         #     bp["boxes"][i].set_facecolor(method_color(tag))
#         #     bp["boxes"][i].set_alpha(0.95)
#
#         # x positions
#         xs = np.arange(1, len(rel_data) + 1)
#         rel_data_clip = []
#         for d in rel_data:
#             d = np.asarray(d)
#             d = d[np.isfinite(d)]
#             if d.size == 0:
#                 rel_data_clip.append(d)
#                 continue
#             lo, hi = np.quantile(d, [0.05, 0.95])
#             rel_data_clip.append(np.clip(d, lo, hi))
#         vp = ax.violinplot(
#             rel_data_clip,
#             positions=xs,
#             widths=0.38,  # 更窄
#             bw_method=0.25,  # 更平滑一点（可试 0.2~0.4）
#             showmeans=False,
#             showmedians=False,  # 我们自己画median/iqr，更好看
#             showextrema=False
#         )
#
#         # 给每个 violin 上色（用你现成的 method_color(tags_used[i])）
#         for i, body in enumerate(vp["bodies"]):
#             body.set_facecolor(method_color(tags_used[i]))
#             body.set_edgecolor("#2B2B2B")
#             body.set_linewidth(1.0)
#             body.set_alpha(0.95)
#         # 画中位数 + IQR（用原始 rel_data 或 clip 都行；建议用原始）
#         for x, d in zip(xs, rel_data):
#             d = np.asarray(d)
#             d = d[np.isfinite(d)]
#             if d.size == 0:
#                 continue
#             q25, q50, q75 = np.quantile(d, [0.25, 0.5, 0.75])
#             # IQR 竖线
#             ax.plot([x, x], [q25, q75], color="#1F1F1F", linewidth=2.0, solid_capstyle="round", zorder=3)
#             # median 横线
#             ax.plot([x - 0.14, x + 0.14], [q50, q50], color="#1F1F1F", linewidth=2.0, zorder=3)
#         # median线样式
#         # vp["cmedians"].set_color("#1F1F1F")
#         # vp["cmedians"].set_linewidth(2.0)
#
#         # x ticks/labels
#         ax.set_xticks(xs)
#         ax.set_xticklabels(labels)
#         ax.axhline(1.0, linestyle="--", linewidth=1.1, color="#666666", alpha=0.9, zorder=0)
#         ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#
#         # ---- 每幅图左上角国家名（你要的）----
#         ax.text(0.02, 0.98, country, transform=ax.transAxes,
#                 ha="left", va="top", fontsize=12, fontweight="bold")
#
#         # ---- 只保留最左列 y tick/label；只保留最下排 x tick/label ----
#         r, c = divmod(idx, 3)
#         if c != 0:
#             ax.set_yticklabels([])
#             ax.set_ylabel("")
#         else:
#             ax.set_ylabel("WIS_model / WIS_Naive")
#
#         if r != 2:
#             ax.set_xticklabels([])
#             ax.set_xlabel("")
#         else:
#             # plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
#             ax.set_xticklabels([])
#
#         # 用于统一ylim（可选）
#         q3s = [np.quantile(d, 0.75) for d in rel_data]
#         all_ymax.append(max(q3s))
#     handles = [mpl.patches.Patch(facecolor=method_color(tag), edgecolor="#2B2B2B", label=label)
#                for tag, label in FIXED_ORDER]
#     fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
#                bbox_to_anchor=(0.5, -0.02), fontsize=10)
#     # ---- ylim：统一（推荐，版面更像你给的例图）----
#     if ylim is not None:
#         y0, y1 = ylim
#     else:
#         ymax = float(np.max(all_ymax)) if all_ymax else 1.5
#         pad = max(0.20 * ymax, 0.30)
#         y0, y1 = 0.0, max(ymax + pad, 1.25)
#
#     for ax in axes:
#         if ax.has_data():
#             ax.set_ylim(y0, y1)
#
#     fig.tight_layout()
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
#     plt.close(fig)
#     print(f"Saved grid: {out_path}")
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

def get_paper_color(tag):
    return PAPER_COLORS.get(tag, "#CCCCCC")


# ==========================================
# 核心绘图函数：水平云雨图 (Horizontal Raincloud)
# ==========================================
def plot_relative_wis_grid_3x3(
        results_root="./results",
        countries=None,
        horizon=1,
        methods=None,
        baseline_tag="Naive_real",
        wis_file_prefix="wis80_point_step",
        settings_contains=None,
        out_path="./test_results/montages/relwis_grid_step1.pdf",
        dpi=800,
        ylim=None,  # 此参数在这里已废弃，改为xlim自适应
):
    _paper_style_rcparams()

    if countries is None:
        countries = DEFAULT_COUNTRIES
    if methods is None:
        methods = DEFAULT_METHODS
    assert len(countries) == 9, "countries 必须正好9个，才能画 3x3"

    run_dirs = _list_run_dirs(results_root, settings_contains=settings_contains)

    # 高度拉长，为了给左侧 Y 轴的 13 个模型标签留出充足呼吸空间
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    axes = axes.ravel()

    # ---- 排序：想要显示在最上面的放前面 ----
    FIXED_ORDER = [
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
    # 反转列表，因为在常规 Y 轴中，坐标越大越靠上
    FIXED_ORDER = FIXED_ORDER[::-1]

    for idx, country in enumerate(countries):
        ax = axes[idx]

        # ---- baseline ----
        base_dir = _find_latest_run(run_dirs, country, baseline_tag)
        if base_dir is None:
            ax.set_axis_off()
            continue
        base_wis, base_dates = _load_wis_from_run(base_dir, wis_file_prefix, horizon)
        if base_wis is None:
            ax.set_axis_off()
            continue
        base_wis = np.where(base_wis == 0, 1e-9, base_wis)

        rel_data = []
        labels = []
        tags_used = []

        for method_tag, method_label in FIXED_ORDER:
            m_dir = _find_latest_run(run_dirs, country, method_tag)
            if m_dir is None:
                continue
            wis, wis_dates = _load_wis_from_run(m_dir, wis_file_prefix, horizon)
            wis_aligned, base_aligned = _align_by_dates(wis, wis_dates, base_wis, base_dates)
            if wis_aligned is None:
                continue

            rel = (wis_aligned / np.where(base_aligned == 0, 1e-9, base_aligned))
            rel = rel[np.isfinite(rel) & (rel > 0.01)]
            if rel.size == 0:
                continue

            rel_data.append(rel)
            labels.append(method_label)
            tags_used.append(method_tag)

        if not rel_data:
            ax.set_axis_off()
            continue

        # Y轴的坐标位置
        ys = np.arange(1, len(rel_data) + 1)

        # ==========================================
        # 1. 绘制“云” (Cloud)：水平方向的半小提琴图
        # ==========================================
        # vp = ax.violinplot(
        #     rel_data,
        #     positions=ys,
        #     vert=False,  # 设置为水平方向
        #     widths=0.8,  # 密度图宽度
        #     bw_method=0.25,
        #     showmeans=False,
        #     showmedians=False,
        #     showextrema=False
        # )

        # offset_cloud = 0.05
        # for i, body in enumerate(vp["bodies"]):
        #     m = ys[i]
        #     path = body.get_paths()[0]
        #     # 水平小提琴中，[:, 1] 是 Y 坐标 (密度)。保留大于中心线的部分，切掉下半部分
        #     path.vertices[:, 1] = np.clip(path.vertices[:, 1], m + offset_cloud, np.inf)
        #
        #     # 色彩调淡：使用 0.45 的透明度
        #     color = get_paper_color(tags_used[i])
        #     body.set_facecolor(color)
        #     body.set_alpha(0.45)
        #     body.set_edgecolor(color)  # 边缘用同色但保持默认不透明度，显得精致
        #     body.set_linewidth(1.0)

        # ==========================================
        # 2. 绘制“核”与“雨” (Box & Rain)
        # ==========================================
        for y_pos, d in zip(ys, rel_data):
            color = get_paper_color(tags_used[y_pos - 1])

            # --- 雨 (Rain)：底部散点抖动 ---
            # 把点往下挪一点，并且随机上下抖动
            y_jitter = y_pos - 0.25 - np.random.uniform(0, 0.25, size=len(d))
            ax.scatter(d, y_jitter, s=6, alpha=0.35, color=color, edgecolors='none', zorder=1)

            # --- 核 (Box)：内嵌横向粗线和白点 ---
            q25, q50, q75 = np.quantile(d, [0.25, 0.5, 0.75])
            box_y = y_pos - 0.08

            # # IQR 粗横线
            # ax.plot([q25, q75], [box_y, box_y], color="#888888", linewidth=1.0, solid_capstyle="round", zorder=3)
            # # Median 白色原点
            # ax.scatter(q50, box_y, color="white", s=18, zorder=4, edgecolor="#888888", linewidth=1.2)
            ax.plot([q25, q75], [box_y, box_y], color=color, linewidth=1.2, solid_capstyle="round", zorder=3)

            ax.scatter(
                q50, box_y,
                facecolors="none",  # 空心
                edgecolors=color,  # 边框用对应模型颜色
                s=18,
                zorder=4,
                linewidth=1.2
            )
        # ==========================================
        # 3. 坐标轴与背景美化
        # ==========================================
        # X轴设为对数(Log2)坐标
        ax.set_xscale('log', base=2)
        x_ticks = [0.25, 0.5, 1.0, 2.0, 4.0]
        ax.set_xticks(x_ticks)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlim(0.2, 5.0)  # 统一限制X轴，保持对比一致性

        # Baseline 参考线 (垂直虚线)
        ax.axvline(1.0, linestyle="--", linewidth=1.2, color="#888888", alpha=0.8, zorder=0)

        # 只保留 X 轴的网格线
        ax.grid(axis="x", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#DDDDDD")  # Y轴线变淡，因为已有Tick

        # 设置 Y 轴标签
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=13)

        # 使用 Title 标注国家名称，看起来更大气
        ax.set_title(country, loc="left", fontsize=14, fontweight="bold", pad=10)

        # ---- 隐藏多余的 Ticks ----
        r, c = divmod(idx, 3)
        # 只有最左边的一列显示 Y 轴模型名
        if c != 0:
            ax.tick_params(axis='y', left=False, labelleft=False)

        # 只有最下面一行显示 X 轴说明
        if r == 2:
            ax.set_xlabel("Relative IS (vs. Naive Real)", fontsize=14, color="#333333", labelpad=8)
        else:
            ax.tick_params(axis='x', bottom=True, labelbottom=False)

    # 稍微拉开子图之间的间距，防止字太挤
    fig.subplots_adjust(wspace=0.15, hspace=0.25)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved horizontal raincloud grid: {out_path}")
if __name__ == "__main__":
    # If you want to filter only certain experiment-name substrings, set settings_contains.
    # Example:
    # settings_contains = ["ili", "uncertainty"]  # ALL tokens must appear
    # settings_contains = "some_run_tag"
    settings_contains = None

    # plot_relative_wis_paper_all(
    #     results_root="./results",
    #     countries=DEFAULT_COUNTRIES,
    #     horizons=(1, 2, 3, 4),
    #     methods=DEFAULT_METHODS,
    #     baseline_tag="Naive_real",
    #     wis_file_prefix="wis80_point_step",
    #     settings_contains=settings_contains,
    #     out_dir="./test_results/montages",
    #     ylim=(0, 4),   # or set e.g. (0.6, 1.6) if you want fixed limits
    # )
    plot_relative_wis_grid_3x3(
        results_root="./results",
        countries=DEFAULT_COUNTRIES,  # 9个国家
        horizon=1,
        methods=DEFAULT_METHODS,
        baseline_tag="Naive_real",
        wis_file_prefix="wis80_point_step",
        settings_contains=None,
        out_path="./test_results/montages/relwis_grid_step1_new.pdf",
        ylim=(0,4),  # 或者 (0,4) 固定
    )