# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


COUNTRIES = [
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

DATASET_DIR = Path("dataset")
OUT_DIR = Path("pics") / "dataset_overview"

KIND_CONFIG = {
    "real": {
        "file_prefix": "real",
        "line_label": "real",
        "out_name": "dataset_real_9countries.png",
        "title": "Real ILI data - all seasons",
    },
    "combined": {
        "file_prefix": "combined",
        "line_label": "comb",
        "out_name": "dataset_combined_9countries.png",
        "title": "Combined ILI data - real plus exogeneous series",
    },
    "augmented": {
        "file_prefix": "augmented",
        "line_label": "aug",
        "out_name": "dataset_augmented_9countries.png",
        "title": "Augmented ILI data - real plus endogeneous series",
    },
}

REQUIRED_COLUMNS = {"item_id", "season_id", "anno", "settimana", "incidenza"}
NPG_COLORS = {
    "real": "#222222",      # charcoal, keeps observed data authoritative
    "combined": "#86BBD8",  # soft blue
    "augmented": "#8FCB9B", # soft green
}
GRID_COLOR = "#ECECEC"
SEASON_DIVIDER_COLOR = "#C7C7C7"
AXIS_COLOR = "#333333"
SYNTHETIC_ALPHA = 0.055
REAL_ALPHA = 0.86


def load_country(kind: str, country: str) -> pd.DataFrame:
    config = KIND_CONFIG[kind]
    path = DATASET_DIR / f"{config['file_prefix']}_{country}_ILI.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df.copy()
    df["item_id"] = df["item_id"].astype(int)
    df["season_id"] = df["season_id"].astype(int)
    df["anno"] = df["anno"].astype(int)
    df["settimana"] = df["settimana"].astype(int)
    df["incidenza"] = df["incidenza"].astype(float)
    return df.sort_values(["item_id", "season_id", "anno", "settimana"])


def build_time_axis(df: pd.DataFrame):
    first_item_id = int(df["item_id"].min())
    ref = df[df["item_id"] == first_item_id].copy()
    ref = ref.sort_values(["season_id", "anno", "settimana"]).reset_index(drop=True)
    ref["_x"] = np.arange(len(ref))

    key_cols = ["season_id", "anno", "settimana"]
    x_lookup = ref.set_index(key_cols)["_x"]

    indexed = df.set_index(key_cols).copy()
    indexed["_x"] = x_lookup
    df_with_x = indexed.reset_index()

    season_spans = []
    for season_id, sub in ref.groupby("season_id", sort=True):
        start = int(sub["_x"].min())
        end = int(sub["_x"].max())
        start_year = int(sub["anno"].min())
        end_year = int(sub["anno"].max())
        if start_year == end_year:
            label = str(start_year)
        else:
            label = f"{start_year}/{str(end_year)[-2:]}"
        season_spans.append((start, end, label))

    return df_with_x, season_spans


def pivot_country(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="item_id",
        columns="_x",
        values="incidenza",
        aggfunc="first",
    )
    return pivot.sort_index().sort_index(axis=1)


def compute_country_y_limits() -> dict[str, tuple[float, float]]:
    y_limits = {}
    for country in COUNTRIES:
        country_max = 0.0
        for kind in KIND_CONFIG:
            df = load_country(kind, country)
            country_max = max(country_max, float(df["incidenza"].max()))
        y_limits[country] = (0.0, country_max * 1.08 if country_max > 0 else 1.0)
    return y_limits


def add_season_axis(ax, season_spans, show_labels: bool):
    for start, _, _ in season_spans[1:]:
        ax.axvline(
            start - 0.5,
            color=SEASON_DIVIDER_COLOR,
            linestyle="--",
            linewidth=0.75,
            alpha=0.75,
            zorder=0,
        )

    mids = [(start + end) / 2 for start, end, _ in season_spans]
    labels = [label for _, _, label in season_spans]
    ax.set_xticks(mids)
    ax.set_xticklabels(labels if show_labels else [])
    if show_labels:
        ax.tick_params(axis="x", labelrotation=0)


def add_series(ax, pivot: pd.DataFrame, kind: str):
    x = pivot.columns.to_numpy(dtype=float)
    first_item_id = int(pivot.index.min())
    real_y = pivot.loc[first_item_id].to_numpy(dtype=float)
    real_color = NPG_COLORS["real"]
    synthetic_color = NPG_COLORS.get(kind, NPG_COLORS["combined"])

    if kind != "real":
        other_ids = [idx for idx in pivot.index if idx != first_item_id]
        if other_ids:
            other_y = pivot.loc[other_ids].to_numpy(dtype=float)
            segments = [np.column_stack([x, y]) for y in other_y]
            synthetic_lines = LineCollection(
                segments,
                colors=synthetic_color,
                linewidths=0.32,
                alpha=SYNTHETIC_ALPHA,
                zorder=1,
            )
            ax.add_collection(synthetic_lines)

    ax.plot(x, real_y, color=real_color, linewidth=1.9, alpha=REAL_ALPHA, zorder=3)


def plot_kind(kind: str, y_limits: dict[str, tuple[float, float]]) -> Path:
    config = KIND_CONFIG[kind]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    axes_flat = axes.ravel()

    for idx, (ax, country) in enumerate(zip(axes_flat, COUNTRIES)):
        df = load_country(kind, country)
        item_count = int(df["item_id"].nunique())
        if kind != "real" and item_count != 1001:
            print(f"[warn] {kind}_{country}_ILI.csv has {item_count} item_ids, not 1001.")

        df, season_spans = build_time_axis(df)
        pivot = pivot_country(df)
        add_series(ax, pivot, kind)

        ax.set_title(country, fontsize=12, pad=6)
        ax.set_ylim(y_limits[country])
        ax.set_xlim(float(pivot.columns.min()), float(pivot.columns.max()))
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(AXIS_COLOR)
        ax.spines["bottom"].set_color(AXIS_COLOR)
        ax.tick_params(colors=AXIS_COLOR)

        row = idx // 3
        col = idx % 3
        add_season_axis(ax, season_spans, show_labels=(row == 2))
        if col == 0:
            ax.set_ylabel("incidenza")

    handles = [Line2D([0], [0], color=NPG_COLORS["real"], linewidth=2.0, label="real")]
    if kind != "real":
        handles.append(Line2D([0], [0], color=NPG_COLORS[kind], linewidth=1.8, label=config["line_label"]))

    fig.suptitle(config["title"], fontsize=15, y=0.985)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / config["out_name"]
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    y_limits = compute_country_y_limits()
    for kind in ["real", "combined", "augmented"]:
        out_path = plot_kind(kind, y_limits)
        print(f"Saved {kind} plot to {out_path}")


if __name__ == "__main__":
    main()
