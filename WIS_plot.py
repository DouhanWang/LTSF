# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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


FAMILY_CMAP = {
    "ARIMA": "Purples",
    "DLinear": "Blues",
    "LSTM": "Greens",
    "Autoformer": "Oranges",
    "TabPFN_ts": "Reds",
    "Ensemble": "Greys",
    "respicast": "YlGnBu",
}


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
            # rel_data after sorting is a list[np.ndarray]
            meds_sorted = np.array([np.median(d) for d in rel_data], dtype=float)

            y0 = 0.0  # can also use 0.6
            stats = []
            for d in rel_data:
                q1, q2, q3 = np.quantile(d, [0.25, 0.5, 0.75])
                stats.append(q3)  #  Q3 represents the upper whisker
            ymax_med = float(np.max(stats))
            pad = max(0.20 * ymax_med, 0.30)
            y1 = max(ymax_med + pad, 1.25)

            # --- plot ---
            fig, ax = plt.subplots(figsize=(10.5, 3.9))

            bp = ax.boxplot(
                rel_data,
                tick_labels=labels,
                widths=0.55,
                showfliers=False,          #  no outliers
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
            

            # baseline reference
            ax.axhline(1.0, linestyle="--", linewidth=1.3, color="#666666", alpha=0.9, zorder=0)

            ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_ylabel("Relative IS (vs. Naive Real)")
            ax.set_ylim(y0, y1)
            ax.margins(y=0.02) 
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


PAPER_COLORS = {

    "ARIMA_real": "#636363",  # Deep Gray
    "SEIR_real": "#008B8B",   # Dark Cyan


    "DLinear_real": "#6BAED6",  # Medium Blue
    "DLinear_aug": "#3182BD",   # Vibrant Blue
    "DLinear_comb": "#08519C",  # Deep Navy


    "LSTM_real": "#74C476",     # Leaf Green
    "LSTM_aug": "#31A354",      # Vibrant Green
    "LSTM_comb": "#006D2C",     # Deep Forest Green


    "Autoformer_real": "#FD8D3C", # Vibrant Orange
    "Autoformer_aug": "#E6550D",  # Rich Burnt Orange
    "Autoformer_comb": "#A63603", # Deep Rust


    "TabPFN_ts_real": "#D81B60",
    "Respicast_real": "#B8860B",
    "Ensemble_real": "#7570B3",
}

def get_paper_color(tag):
    return PAPER_COLORS.get(tag, "#CCCCCC")


# ==========================================
#(Horizontal Raincloud)
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
        ylim=None,  
):
    _paper_style_rcparams()

    if countries is None:
        countries = DEFAULT_COUNTRIES
    if methods is None:
        methods = DEFAULT_METHODS
    assert len(countries) == 9, "countries must be exactly 9 to create a 3x3 grid"

    run_dirs = _list_run_dirs(results_root, settings_contains=settings_contains)


    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    axes = axes.ravel()

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
        ("Ensemble_real", "Ensemble"),
        ("Respicast_real", "RespiCast"),
    ]

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


        ys = np.arange(1, len(rel_data) + 1)


        # ==========================================
        # 2. (Box & Rain)
        # ==========================================
        for y_pos, d in zip(ys, rel_data):
            color = get_paper_color(tags_used[y_pos - 1])

            # --- (Rain)---
            y_jitter = y_pos - 0.25 - np.random.uniform(0, 0.25, size=len(d))
            ax.scatter(d, y_jitter, s=6, alpha=0.35, color=color, edgecolors='none', zorder=1)

            # --- (Box) ---
            q25, q50, q75 = np.quantile(d, [0.25, 0.5, 0.75])
            box_y = y_pos - 0.08


            ax.plot([q25, q75], [box_y, box_y], color=color, linewidth=1.2, solid_capstyle="round", zorder=3)

            ax.scatter(
                q50, box_y,
                facecolors="none",
                edgecolors=color,
                s=18,
                zorder=4,
                linewidth=1.2
            )

        ax.set_xscale('log', base=2)
        x_ticks = [0.25, 0.5, 1.0, 2.0, 4.0]
        ax.set_xticks(x_ticks)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlim(0.2, 5.0)

        # Baseline
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
            ax.set_xlabel("Relative IS (vs. Naive Real)", fontsize=14, color="#333333", labelpad=8)
        else:
            ax.tick_params(axis='x', bottom=True, labelbottom=False)


    fig.subplots_adjust(wspace=0.15, hspace=0.25)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved horizontal raincloud grid: {out_path}")
if __name__ == "__main__":

    settings_contains = None

    plot_relative_wis_grid_3x3(
        results_root="./results",
        countries=DEFAULT_COUNTRIES,
        horizon=1,
        methods=DEFAULT_METHODS,
        baseline_tag="Naive_real",
        wis_file_prefix="wis80_point_step",
        settings_contains=None,
        out_path="./test_results/montages/relwis_grid_step1_new.pdf",
        ylim=(0,4),
    )