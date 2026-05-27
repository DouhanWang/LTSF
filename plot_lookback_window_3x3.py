# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd


DEFAULT_COUNTRIES = [
    "Belgium",
    "Czechia",
    "Denmark",
    "France",
    "Ireland",
    "Italy",
    "Netherlands",
    "Poland",
    "Romania",
]
DEFAULT_MODELS = ["DLinear", "LSTM", "Autoformer"]
DEFAULT_LOOKBACKS = [4, 6, 8]
DEFAULT_SETTINGS = ["real", "augmented", "combined"]

SETTING_LABELS = {
    "real": "Real",
    "augmented": "Aug",
    "combined": "Comb",
}

PAPER_COLORS = {
    "DLinear_real": "#6BAED6",
    "DLinear_augmented": "#3182BD",
    "DLinear_combined": "#08519C",
    "LSTM_real": "#74C476",
    "LSTM_augmented": "#31A354",
    "LSTM_combined": "#006D2C",
    "Autoformer_real": "#FD8D3C",
    "Autoformer_augmented": "#E6550D",
    "Autoformer_combined": "#A63603",
}

COUNTRY_YLIM = {
    "Belgium": (0, 2000),
    "Czechia": (0, 450),
    "Denmark": (0, 500),
    "France": (0, 1200),
    "Ireland": (0, 200),
    "Italy": (0, 3000),
    "Netherlands": (0, 300),
    "Poland": (0, 1800),
    "Romania": (0, 100),
}


def soften_color(hex_color: str, amount: float = 0.16) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(hex_color))
    return tuple(rgb + (1.0 - rgb) * amount)


def get_prediction_color(model: str, setting: str) -> tuple[float, float, float]:
    return soften_color(PAPER_COLORS.get(f"{model}_{setting}", "#333333"))


def get_prediction_label(model: str, setting: str) -> str:
    return f"{model} ({SETTING_LABELS.get(setting, setting).lower()})"


@dataclass
class ResultSeries:
    folder: Path
    dates: pd.Series
    true: pd.Series
    pred: pd.Series
    lower: pd.Series | None
    upper: pd.Series | None


def parse_dates(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, format="mixed", dayfirst=True, errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(values, dayfirst=True, errors="coerce")


def result_folder_candidates(results_dir: Path, model: str, setting: str, country: str, lookback: int) -> list[Path]:
    base = results_dir / f"{model}_{setting}_{country}_ILI"

    if lookback == 4:
        return [
            base,
            Path(f"{base}_4"),
            Path(f"{base}_seq4"),
            Path(f"{base}_lookback4"),
        ]

    return [
        Path(f"{base}_{lookback}"),
        Path(f"{base}_seq{lookback}"),
        Path(f"{base}_lookback{lookback}"),
    ]


def find_result_folder(results_dir: Path, model: str, setting: str, country: str, lookback: int) -> Path | None:
    for folder in result_folder_candidates(results_dir, model, setting, country, lookback):
        if folder.exists():
            return folder
    return None


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    by_lower = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def first_column_containing(df: pd.DataFrame, needles: tuple[str, ...]) -> str | None:
    for col in df.columns:
        col_lower = str(col).lower()
        if any(needle in col_lower for needle in needles):
            return col
    return None


def load_result(results_dir: Path, model: str, setting: str, country: str, lookback: int, step: int) -> ResultSeries | None:
    folder = find_result_folder(results_dir, model, setting, country, lookback)
    if folder is None:
        return None

    csv_path = folder / f"rolling_pred_step{step}.csv"
    if not csv_path.exists():
        csv_path = folder / f"rolling_pred_step_{step}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    date_col = first_existing_column(df, ["date", "time", "timestamp"])
    true_col = first_existing_column(df, ["true", "truth", "ground_truth", "incidenza", "target"])
    pred_col = first_existing_column(df, [f"pred_step{step}", "pred", "y_pred", "forecast", "median", "0.5"])
    lower_col = first_existing_column(df, [f"lower80_step{step}", "lower", "lower80", "0.1"])
    upper_col = first_existing_column(df, [f"upper80_step{step}", "upper", "upper80", "0.9"])

    if lower_col is None:
        lower_col = first_column_containing(df, ("lower", "lo"))
    if upper_col is None:
        upper_col = first_column_containing(df, ("upper", "hi"))
    if date_col is None or true_col is None or pred_col is None:
        return None

    dates = parse_dates(df[date_col])
    true = pd.to_numeric(df[true_col], errors="coerce")
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    lower = pd.to_numeric(df[lower_col], errors="coerce") if lower_col else None
    upper = pd.to_numeric(df[upper_col], errors="coerce") if upper_col else None

    return ResultSeries(folder=folder, dates=dates, true=true, pred=pred, lower=lower, upper=upper)


def load_dataset_truth(dataset_dir: Path, country: str) -> tuple[pd.Series, pd.Series] | None:
    csv_path = dataset_dir / f"real_{country}_ILI.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if "season_id" in df.columns:
        df = df[df["season_id"] == df["season_id"].max()].copy()

    if {"anno", "settimana", "incidenza"}.issubset(df.columns):
        date_str = df["anno"].astype(str) + " " + df["settimana"].astype(str) + " 1"
        dates = pd.to_datetime(date_str, format="%Y %W %w", errors="coerce")
        order = np.argsort(dates.to_numpy())
        dates = dates.iloc[order].reset_index(drop=True)
        true = pd.to_numeric(df["incidenza"], errors="coerce").iloc[order].reset_index(drop=True)
        return dates, true

    return None


def choose_truth(results: list[ResultSeries | None], dataset_dir: Path, country: str) -> tuple[pd.Series, pd.Series] | None:
    for result in results:
        if result is not None and result.true.notna().any():
            return result.dates.reset_index(drop=True), result.true.reset_index(drop=True)
    return load_dataset_truth(dataset_dir, country)


def date_index_lookup(dates: pd.Series) -> dict[pd.Timestamp, int]:
    lookup = {}
    for i, value in enumerate(dates):
        if pd.notna(value):
            lookup[pd.Timestamp(value).normalize()] = i
    return lookup


def align_to_truth(result: ResultSeries, truth_dates: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    lookup = date_index_lookup(truth_dates)
    x_vals = []
    pred_vals = []
    lower_vals = []
    upper_vals = []

    for i, value in enumerate(result.pred):
        if not np.isfinite(value):
            continue

        x = None
        if i < len(result.dates) and pd.notna(result.dates.iloc[i]):
            x = lookup.get(pd.Timestamp(result.dates.iloc[i]).normalize())
        if x is None and i < len(truth_dates):
            x = i
        if x is None:
            continue

        x_vals.append(x)
        pred_vals.append(float(value))

        if result.lower is not None and i < len(result.lower) and np.isfinite(result.lower.iloc[i]):
            lower_vals.append(float(result.lower.iloc[i]))
        else:
            lower_vals.append(np.nan)

        if result.upper is not None and i < len(result.upper) and np.isfinite(result.upper.iloc[i]):
            upper_vals.append(float(result.upper.iloc[i]))
        else:
            upper_vals.append(np.nan)

    x_arr = np.asarray(x_vals, dtype=float)
    pred_arr = np.asarray(pred_vals, dtype=float)
    lower_arr = np.asarray(lower_vals, dtype=float) if lower_vals else None
    upper_arr = np.asarray(upper_vals, dtype=float) if upper_vals else None

    return x_arr, pred_arr, lower_arr, upper_arr


def style_axes(ax: plt.Axes, country: str, is_bottom: bool, truth_dates: pd.Series) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_xlim(-0.5, max(len(truth_dates) - 0.5, 0.5))

    if country in COUNTRY_YLIM:
        ax.set_ylim(COUNTRY_YLIM[country])

    tick_idx = np.linspace(0, len(truth_dates) - 1, min(6, len(truth_dates)), dtype=int)
    ax.set_xticks(tick_idx)

    if is_bottom:
        labels = []
        for i in tick_idx:
            value = truth_dates.iloc[i]
            labels.append(value.strftime("%Y-%m-%d") if pd.notna(value) else "")
        ax.set_xticklabels(labels, rotation=30, ha="right")
    else:
        ax.tick_params(axis="x", labelbottom=False)


def plot_country(
    country: str,
    models: list[str],
    lookbacks: list[int],
    settings: list[str],
    results_dir: Path,
    dataset_dir: Path,
    out_dir: Path,
    step: int,
    show_intervals: bool,
) -> Path:
    fig, axes = plt.subplots(len(lookbacks), len(settings), figsize=(15, 10), sharey=True)
    axes = np.atleast_2d(axes)
    legend_handles: dict[str, mlines.Line2D] = {}

    for row, lookback in enumerate(lookbacks):
        for col, setting in enumerate(settings):
            ax = axes[row, col]
            cell_results = [
                load_result(results_dir, model, setting, country, lookback, step)
                for model in models
            ]
            truth = choose_truth(cell_results, dataset_dir, country)

            if truth is None:
                ax.text(0.5, 0.5, "Missing ground truth", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            truth_dates, truth_values = truth
            x_true = np.arange(len(truth_values))
            truth_line, = ax.plot(
                x_true,
                truth_values,
                color="black",
                marker="o",
                markersize=3.2,
                linestyle="None",
                label="Ground truth",
                zorder=50,
            )
            legend_handles.setdefault("Ground truth", truth_line)

            ax.axvline(x=lookback - 1, color="#777777", linestyle=":", linewidth=1.3, alpha=0.8, zorder=45)
            split_handle = mlines.Line2D([], [], color="#777777", linestyle=":", linewidth=1.6, label="Lookback end")
            legend_handles.setdefault("Lookback end", split_handle)

            plotted_any = False
            for model, result in zip(models, cell_results):
                if result is None:
                    continue

                x_pred, pred, lower, upper = align_to_truth(result, truth_dates)
                if len(pred) == 0:
                    continue

                plotted_any = True
                color = get_prediction_color(model, setting)
                label = get_prediction_label(model, setting)
                line, = ax.plot(x_pred, pred, color=color, linewidth=2.0, label=label, zorder=20)
                legend_handles.setdefault(label, line)

                if show_intervals and lower is not None and upper is not None:
                    valid = np.isfinite(x_pred) & np.isfinite(pred) & np.isfinite(lower) & np.isfinite(upper)
                    if valid.any():
                        ax.fill_between(
                            x_pred[valid],
                            lower[valid],
                            upper[valid],
                            color=color,
                            alpha=0.18,
                            linewidth=0,
                            zorder=10,
                        )

            if not plotted_any:
                ax.text(0.5, 0.86, "Missing predictions", ha="center", va="center", transform=ax.transAxes, color="#777777")

            if row == 0:
                ax.set_title(SETTING_LABELS.get(setting, setting), fontsize=13, fontweight="bold")

            if col == 0:
                ax.set_ylabel(f"Lookback={lookback}\nIncidence", fontsize=12)

            style_axes(ax, country, row == len(lookbacks) - 1, truth_dates)

    if len(models) == 1:
        title = f"{country} - {models[0]} look-back-window comparison (horizon step {step})"
    else:
        title = f"{country} look-back-window comparison (horizon step {step})"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.legend(
        handles=list(legend_handles.values()),
        loc="lower center",
        ncol=min(5, len(legend_handles)),
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=11,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, wspace=0.12, hspace=0.25)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = "all_models" if len(models) > 1 else models[0]
    out_path = out_dir / f"{country}_lookback_3x3_step{step}_{model_tag}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 3x3 look-back-window forecast comparisons by country.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--out-dir", default="test_results/lookback_window_3x3")
    parser.add_argument("--step", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--countries", nargs="*", default=DEFAULT_COUNTRIES)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--lookbacks", nargs="*", type=int, default=DEFAULT_LOOKBACKS)
    parser.add_argument("--settings", nargs="*", default=DEFAULT_SETTINGS)
    parser.add_argument("--hide-intervals", action="store_true", help="Do not draw prediction intervals.")
    parser.add_argument("--show-intervals", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--overlay-models",
        action="store_true",
        help="Overlay all selected models in one 3x3 figure per country.",
    )
    parser.add_argument(
        "--split-models",
        action="store_true",
        help="Backward-compatible alias; per-model 3x3 figures are now the default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)

    generated = []
    split_by_model = args.split_models or not args.overlay_models
    show_intervals = (not args.hide_intervals) or args.show_intervals

    if split_by_model:
        for model in args.models:
            model_out_dir = out_dir / model
            for country in args.countries:
                generated.append(
                    plot_country(
                        country=country,
                        models=[model],
                        lookbacks=args.lookbacks,
                        settings=args.settings,
                        results_dir=results_dir,
                        dataset_dir=dataset_dir,
                        out_dir=model_out_dir,
                        step=args.step,
                        show_intervals=show_intervals,
                    )
                )
    else:
        for country in args.countries:
            generated.append(
                plot_country(
                    country=country,
                    models=args.models,
                    lookbacks=args.lookbacks,
                    settings=args.settings,
                    results_dir=results_dir,
                    dataset_dir=dataset_dir,
                    out_dir=out_dir,
                    step=args.step,
                    show_intervals=show_intervals,
                )
            )

    for path in generated:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
