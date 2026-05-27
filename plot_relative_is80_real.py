import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from WIS_plot import (
    DEFAULT_COUNTRIES,
    _align_by_dates,
    _load_wis_from_run,
    _paper_style_rcparams,
    get_paper_color,
)


REAL_ONLY_METHODS = [
    ("ARIMA_real", "ARIMA"),
    ("SEIR_real", "SEIR"),
    ("DLinear_real", "DLinear (real)"),
    ("LSTM_real", "LSTM (real)"),
    ("Autoformer_real", "Autoformer (real)"),
    ("TabPFN_ts_real", "TabPFN-TS"),
    ("Ensemble_real", "Ensemble"),
    ("Respicast_real", "RespiCast"),
]

WINDOWED_REAL_TAGS = {"DLinear_real", "LSTM_real", "Autoformer_real"}


def _window_suffix(window):
    if window is None or str(window).strip().lower() in {"", "none", "full"}:
        return ""
    return f"_{int(window)}"


def _tag_to_dir_name(country, method_tag, window=None):
    suffix = _window_suffix(window) if method_tag in WINDOWED_REAL_TAGS else ""
    return f"{method_tag}_{country}_ILI{suffix}"


def _get_run_dir(results_root, country, method_tag, window=None):
    run_dir = os.path.join(results_root, _tag_to_dir_name(country, method_tag, window))
    if os.path.isdir(run_dir):
        return run_dir
    return None


def _load_relative_is80(
    results_root,
    country,
    method_tag,
    horizon,
    baseline_tag="Naive_real",
    wis_file_prefix="wis80_point_step",
    window=None,
):
    base_dir = _get_run_dir(results_root, country, baseline_tag, window=None)
    if base_dir is None:
        print(f"[skip] {country} step{horizon}: missing baseline {baseline_tag}")
        return None

    base_is, base_dates = _load_wis_from_run(base_dir, wis_file_prefix, horizon)
    if base_is is None:
        print(f"[skip] {country} step{horizon}: missing baseline IS80 in {base_dir}")
        return None

    method_dir = _get_run_dir(results_root, country, method_tag, window=window)
    if method_dir is None:
        print(f"[skip] {country} step{horizon}: missing run for {method_tag}")
        return None

    is80, is80_dates = _load_wis_from_run(method_dir, wis_file_prefix, horizon)
    is80_aligned, base_aligned = _align_by_dates(is80, is80_dates, base_is, base_dates)
    if is80_aligned is None:
        print(f"[skip] {country} step{horizon}: no common dates for {method_tag}")
        return None

    base_aligned = np.where(base_aligned == 0, 1e-9, base_aligned)
    rel = is80_aligned / base_aligned
    rel = rel[np.isfinite(rel) & (rel > 0.01)]
    if rel.size == 0:
        print(f"[skip] {country} step{horizon}: empty relative IS80 for {method_tag}")
        return None
    return rel


def _plot_country_panel(
    ax,
    country,
    methods,
    results_root,
    horizon,
    baseline_tag,
    wis_file_prefix,
    window,
    rng,
):
    rel_data = []
    labels = []
    tags_used = []

    for method_tag, method_label in methods[::-1]:
        rel = _load_relative_is80(
            results_root=results_root,
            country=country,
            method_tag=method_tag,
            horizon=horizon,
            baseline_tag=baseline_tag,
            wis_file_prefix=wis_file_prefix,
            window=window,
        )
        if rel is None:
            continue
        rel_data.append(rel)
        labels.append(method_label)
        tags_used.append(method_tag)

    if not rel_data:
        ax.set_axis_off()
        return False

    ys = np.arange(1, len(rel_data) + 1)
    for y_pos, values, tag in zip(ys, rel_data, tags_used):
        color = get_paper_color(tag)
        y_jitter = y_pos - 0.25 - rng.uniform(0, 0.25, size=len(values))
        ax.scatter(values, y_jitter, s=6, alpha=0.35, color=color, edgecolors="none", zorder=1)

        q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        box_y = y_pos - 0.08
        ax.plot([q25, q75], [box_y, box_y], color=color, linewidth=1.2, solid_capstyle="round", zorder=3)
        ax.scatter(
            q50,
            box_y,
            facecolors="none",
            edgecolors=color,
            s=18,
            zorder=4,
            linewidth=1.2,
        )

    ax.set_xscale("log", base=2)
    x_ticks = [0.25, 0.5, 1.0, 2.0, 4.0]
    ax.set_xticks(x_ticks)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(0.2, 5.0)

    ax.axvline(1.0, linestyle="--", linewidth=1.2, color="#888888", alpha=0.8, zorder=0)
    ax.grid(axis="x", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_title(country, loc="left", fontsize=14, fontweight="bold", pad=10)
    return True


def plot_relative_is80_real(
    results_root="./results",
    countries=None,
    horizon=4,
    methods=None,
    baseline_tag="Naive_real",
    wis_file_prefix="wis80_point_step",
    out_path="./test_results/montages/relis80_real_grid_step4.png",
    dpi=800,
    window=None,
    seed=2026,
):
    _paper_style_rcparams()

    if countries is None:
        countries = DEFAULT_COUNTRIES
    if methods is None:
        methods = REAL_ONLY_METHODS

    rng = np.random.default_rng(seed)
    n = len(countries)
    if n == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7.0, 4.2))
        axes = np.array([axes])
        nrows, ncols = 1, 1
    else:
        ncols = 3
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.7 * nrows), sharex=True, sharey=True)
        axes = np.asarray(axes).ravel()

    for idx, country in enumerate(countries):
        ax = axes[idx]
        plotted = _plot_country_panel(
            ax=ax,
            country=country,
            methods=methods,
            results_root=results_root,
            horizon=horizon,
            baseline_tag=baseline_tag,
            wis_file_prefix=wis_file_prefix,
            window=window,
            rng=rng,
        )
        if not plotted:
            continue

        row, col = divmod(idx, ncols)
        if n > 1 and col != 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
        if row == nrows - 1:
            ax.set_xlabel("Relative IS80 (vs. Naive Real)", fontsize=14, color="#333333", labelpad=8)
        elif n > 1:
            ax.tick_params(axis="x", bottom=True, labelbottom=False)

    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    if n == 1:
        fig.tight_layout()
    else:
        fig.subplots_adjust(wspace=0.15, hspace=0.25)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved relative IS80 real-only plot: {out_path}")


def _parse_countries(value):
    if value is None or value.strip().lower() == "all":
        return DEFAULT_COUNTRIES
    return [x.strip() for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Plot relative IS80 using only real methods, without augmented or combined models."
    )
    parser.add_argument("--results-root", default="./results")
    parser.add_argument("--countries", default="all", help='Use "all" or a comma-separated list, e.g. France.')
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument(
        "--window",
        default=None,
        help="Use None/full for directories like *_France_ILI, or 6/8 for deep-model real directories.",
    )
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--dpi", type=int, default=800)
    args = parser.parse_args()

    countries = _parse_countries(args.countries)
    window_label = "full" if args.window is None else str(args.window).lower()
    if args.out_path is None:
        scope = "grid" if len(countries) > 1 else countries[0].lower()
        out_path = f"./test_results/montages/relis80_real_{scope}_step{args.horizon}_{window_label}.png"
    else:
        out_path = args.out_path

    plot_relative_is80_real(
        results_root=args.results_root,
        countries=countries,
        horizon=args.horizon,
        out_path=out_path,
        dpi=args.dpi,
        window=args.window,
    )


if __name__ == "__main__":
    main()
