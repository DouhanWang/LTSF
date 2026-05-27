import argparse
import os

import matplotlib
import matplotlib as mpl
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rAE_plot import (
    _align_by_dates,
    _load_data_from_csv,
    _paper_style_rcparams,
    get_paper_color,
)


COUNTRY = "France"

REAL_METHODS = [
    ("LSTM_real", "LSTM (real)"),
    ("DLinear_real", "DLinear (real)"),
    ("Autoformer_real", "Autoformer (real)"),
]

AUG_COMB_METHODS = [
    ("LSTM_aug", "LSTM (aug)"),
    ("DLinear_aug", "DLinear (aug)"),
    ("Autoformer_aug", "Autoformer (aug)"),
    ("LSTM_comb", "LSTM (comb)"),
    ("DLinear_comb", "DLinear (comb)"),
    ("Autoformer_comb", "Autoformer (comb)"),
]

VARIANT_DIR_NAMES = {
    "real": "real",
    "aug": "augmented",
    "comb": "combined",
}


def _window_suffix(window):
    if window is None or str(window).strip().lower() in {"", "none", "full"}:
        return ""
    return f"_{int(window)}"


def _tag_to_dir_name(country, method_tag, window=None):
    model, variant = method_tag.split("_", 1)
    variant_dir = VARIANT_DIR_NAMES.get(variant, variant)
    return f"{model}_{variant_dir}_{country}_ILI{_window_suffix(window)}"


def _get_run_dir(results_root, country, method_tag, window=None):
    run_dir = os.path.join(results_root, _tag_to_dir_name(country, method_tag, window))
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Missing result directory: {run_dir}")
    return run_dir


def _relative_absolute_errors(
    results_root,
    country,
    method_tag,
    baseline_tag="Naive_real",
    horizon=1,
    window=None,
):
    base_dir = _get_run_dir(results_root, country, baseline_tag, window=None)
    base_pred, base_true, base_dates = _load_data_from_csv(base_dir, baseline_tag, horizon)
    if base_pred is None or base_true is None:
        raise ValueError(f"Cannot load baseline data from {base_dir}")

    method_dir = _get_run_dir(results_root, country, method_tag, window=window)
    pred, true, dates = _load_data_from_csv(method_dir, method_tag, horizon)
    if pred is None:
        raise ValueError(f"Cannot load prediction data from {method_dir}")

    if true is None:
        if base_dates is None or dates is None:
            raise ValueError(f"{method_dir} has no true values and cannot be date-aligned.")
        true_by_date = {str(d): v for d, v in zip(base_dates, base_true)}
        true = np.array([true_by_date.get(str(d), np.nan) for d in dates], dtype=float)

    model_ae = np.abs(np.asarray(pred, dtype=float) - np.asarray(true, dtype=float))
    base_ae = np.abs(np.asarray(base_pred, dtype=float) - np.asarray(base_true, dtype=float))

    model_ae, base_ae = _align_by_dates(model_ae, dates, base_ae, base_dates)
    if model_ae is None or base_ae is None:
        raise ValueError(f"Cannot align {method_tag} with {baseline_tag}.")

    base_ae = np.where(base_ae == 0, 1e-9, base_ae)
    rel = model_ae / base_ae
    return rel[np.isfinite(rel) & (rel > 0.01)]


def plot_france_rae(
    methods,
    out_path,
    results_root="./results",
    country=COUNTRY,
    horizon=1,
    window=None,
    baseline_tag="Naive_real",
    dpi=800,
    seed=2026,
):
    _paper_style_rcparams()

    rel_data = []
    labels = []
    tags = []
    for tag, label in methods:
        rel = _relative_absolute_errors(
            results_root=results_root,
            country=country,
            method_tag=tag,
            baseline_tag=baseline_tag,
            horizon=horizon,
            window=window,
        )
        if rel.size == 0:
            print(f"[warn] {country} {tag} has no valid RAE values, skipped.")
            continue
        rel_data.append(rel)
        labels.append(label)
        tags.append(tag)

    if not rel_data:
        raise RuntimeError("No valid RAE values were loaded.")

    rng = np.random.default_rng(seed)
    height = max(2.6, 0.55 * len(rel_data) + 1.0)
    fig, ax = plt.subplots(figsize=(7.0, height))

    ys = np.arange(1, len(rel_data) + 1)
    for y_pos, values, tag in zip(ys, rel_data, tags):
        color = get_paper_color(tag)
        y_jitter = y_pos - 0.25 - rng.uniform(0, 0.25, size=len(values))
        ax.scatter(
            values,
            y_jitter,
            s=6,
            alpha=0.35,
            color=color,
            edgecolors="none",
            zorder=1,
        )

        q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        box_y = y_pos - 0.08
        ax.plot(
            [q25, q75],
            [box_y, box_y],
            color=color,
            linewidth=1.2,
            solid_capstyle="round",
            zorder=3,
        )
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
    ax.set_xlabel("Relative Absolute Error (vs. Naive Real)", fontsize=14, color="#333333", labelpad=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot two France RAE figures using the existing rAE_plot.py colors and style."
    )
    parser.add_argument("--results-root", default="./results")
    parser.add_argument("--out-dir", default="./test_results/montages")
    parser.add_argument("--country", default=COUNTRY)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--window",
        default=None,
        help="Use None/full for directories like *_France_ILI, or 6/8 for *_France_ILI_6/8.",
    )
    parser.add_argument("--dpi", type=int, default=800)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Output file formats.",
    )
    args = parser.parse_args()

    window_label = "full" if args.window is None else str(args.window).lower()
    for fmt in args.formats:
        real_out = os.path.join(args.out_dir, f"france_rae_real_step{args.horizon}_{window_label}.{fmt}")
        aug_comb_out = os.path.join(args.out_dir, f"france_rae_aug_comb_step{args.horizon}_{window_label}.{fmt}")

        plot_france_rae(
            methods=REAL_METHODS,
            out_path=real_out,
            results_root=args.results_root,
            country=args.country,
            horizon=args.horizon,
            window=args.window,
            dpi=args.dpi,
        )
        plot_france_rae(
            methods=AUG_COMB_METHODS,
            out_path=aug_comb_out,
            results_root=args.results_root,
            country=args.country,
            horizon=args.horizon,
            window=args.window,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
