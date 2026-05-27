# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
OUT_DIR = Path("results/metrics_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["Belgium", "Czechia", "Denmark", "France", "Ireland", "Italy", "Netherlands", "Poland", "Romania"]
MODELS = ["DLinear", "LSTM", "Autoformer"]
TRAIN_SETTINGS = ["real", "augmented", "combined"]
LOOKBACKS = [4, 6, 8]
STEPS = [1, 2, 3, 4]

METRICS = [
    ("MAE", "MAE"),
    ("WMAPE", "WMAPE"),
    ("WIS80", "IS80"),
]

SETTING_LABELS = {
    "real": "real",
    "augmented": "aug",
    "combined": "comb",
}


def result_folder(model: str, setting: str, country: str, lookback: int) -> Path:
    base = RESULTS_DIR / f"{model}_{setting}_{country}_ILI"
    if lookback == 4:
        return base
    return Path(f"{base}_{lookback}")


def parse_metrics_file(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}

    latest_by_step: dict[int, dict[str, float]] = {}
    token_re = re.compile(r"([A-Za-z0-9_]+):([-+0-9.eE]+)")

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = {key: float(value) for key, value in token_re.findall(line)}
        if "step" not in tokens:
            continue
        step = int(round(tokens["step"]))
        latest_by_step[step] = tokens

    return latest_by_step


def collect_records() -> pd.DataFrame:
    records = []

    for model in MODELS:
        for setting in TRAIN_SETTINGS:
            for country in COUNTRIES:
                for lookback in LOOKBACKS:
                    folder = result_folder(model, setting, country, lookback)
                    metrics_by_step = parse_metrics_file(folder / "metrics_by_step.txt")

                    for step in STEPS:
                        row = {
                            "country": country,
                            "step": step,
                            "method": model,
                            "train_setting": setting,
                            "lookback": lookback,
                            "result_folder": folder.as_posix(),
                        }

                        metrics = metrics_by_step.get(step, {})
                        for metric_key, _ in METRICS:
                            row[metric_key] = metrics.get(metric_key, np.nan)
                        records.append(row)

    return pd.DataFrame(records)


def fmt(value: float | int | None, decimals: int | None = None) -> str:
    if value is None or pd.isna(value) or np.isinf(float(value)):
        return ""
    if decimals is not None:
        return f"{float(value):.{decimals}f}"
    return f"{float(value):.4g}"


def metric_decimals(metric_key: str) -> int | None:
    if metric_key == "WMAPE":
        return 3
    return None


def escape_latex(text: str) -> str:
    return str(text).replace("_", r"\_")


def build_full_col_plan() -> list[tuple[str, str, int]]:
    return [(model, setting, lookback) for model in MODELS for setting in TRAIN_SETTINGS for lookback in LOOKBACKS]


def cmidrules_for_groups(group_sizes: list[int], start_col: int = 3) -> str:
    parts = []
    col = start_col
    for size in group_sizes:
        parts.append(rf"\cmidrule(lr){{{col}-{col + size - 1}}}")
        col += size
    return " ".join(parts)


def make_full_metric_table(df: pd.DataFrame, metric_key: str, metric_display: str) -> Path:
    decimals = metric_decimals(metric_key)
    col_plan = build_full_col_plan()
    row_index = pd.MultiIndex.from_tuples(
        [(country, step) for country in COUNTRIES for step in STEPS],
        names=["country", "step"],
    )
    col_index = pd.MultiIndex.from_tuples(col_plan, names=["method", "train_setting", "lookback"])

    wide = df.pivot_table(
        index=["country", "step"],
        columns=["method", "train_setting", "lookback"],
        values=metric_key,
        aggfunc="first",
    ).reindex(index=row_index, columns=col_index)

    display = wide.astype(object)
    for idx in wide.index:
        for model in MODELS:
            for setting in TRAIN_SETTINGS:
                cols = [(model, setting, lookback) for lookback in LOOKBACKS]
                vals = wide.loc[idx, cols]
                best = vals.min(skipna=True)
                for col in cols:
                    value = wide.loc[idx, col]
                    if pd.isna(value):
                        display.loc[idx, col] = ""
                    else:
                        value_text = fmt(value, decimals=decimals)
                        display.loc[idx, col] = (
                            rf"\textbf{{{value_text}}}"
                            if pd.notna(best) and np.isclose(float(value), float(best))
                            else value_text
                        )

    h1 = [r"\multirow{3}{*}{\textbf{Country}}", r"\multirow{3}{*}{\textbf{Step}}"]
    for model in MODELS:
        h1.append(rf"\multicolumn{{9}}{{c}}{{\textbf{{{escape_latex(model)}}}}}")

    h2 = ["", ""]
    for _model in MODELS:
        for setting in TRAIN_SETTINGS:
            h2.append(rf"\multicolumn{{3}}{{c}}{{{SETTING_LABELS[setting]}}}")

    h3 = ["", ""]
    for _model in MODELS:
        for _setting in TRAIN_SETTINGS:
            h3.extend([f"k={lookback}" for lookback in LOOKBACKS])

    colspec = "ll " + " ".join(["ccc"] * (len(MODELS) * len(TRAIN_SETTINGS)))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\fontsize{4.8pt}{6pt}\selectfont",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(h1) + r" \\",
        cmidrules_for_groups([9, 9, 9]),
        " & ".join(h2) + r" \\",
        cmidrules_for_groups([3] * (len(MODELS) * len(TRAIN_SETTINGS))),
        " & ".join(h3) + r" \\",
        r"\midrule",
    ]

    country_span = len(STEPS)
    values = display.to_numpy()
    for row_i, (country, step) in enumerate(display.index.tolist()):
        country_cell = rf"\multirow{{{country_span}}}{{*}}{{{escape_latex(country)}}}" if row_i % country_span == 0 else ""
        lines.append(" & ".join([country_cell, str(step)] + list(values[row_i])) + r" \\")
        if (row_i + 1) % country_span == 0 and (row_i + 1) != len(display):
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\caption{{Deep learning look-back-window comparison for {metric_display}. "
            r"Bold indicates the best look-back window within each model and training setting.}}",
            rf"\label{{tab:deep-window-full-{metric_display.lower()}}}",
            r"\end{table}",
        ]
    )

    out_path = OUT_DIR / f"deep_window_full_{metric_display}.tex"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def make_summary_table(df: pd.DataFrame, metric_key: str, metric_display: str) -> Path:
    decimals = metric_decimals(metric_key)
    summary = (
        df.groupby(["method", "train_setting", "lookback"], dropna=False)[metric_key]
        .mean()
        .reset_index()
        .pivot_table(index=["method", "train_setting"], columns="lookback", values=metric_key, aggfunc="first")
        .reindex(
            pd.MultiIndex.from_tuples(
                [(model, setting) for model in MODELS for setting in TRAIN_SETTINGS],
                names=["method", "train_setting"],
            )
        )
        .reindex(columns=LOOKBACKS)
    )

    display = summary.astype(object)
    for idx in summary.index:
        best = summary.loc[idx].min(skipna=True)
        for lookback in LOOKBACKS:
            value = summary.loc[idx, lookback]
            if pd.isna(value):
                display.loc[idx, lookback] = ""
            else:
                value_text = fmt(value, decimals=decimals)
                display.loc[idx, lookback] = (
                    rf"\textbf{{{value_text}}}"
                    if pd.notna(best) and np.isclose(float(value), float(best))
                    else value_text
                )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\fontsize{8pt}{10pt}\selectfont",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Train} & \textbf{k=4} & \textbf{k=6} & \textbf{k=8} \\",
        r"\midrule",
    ]

    for model in MODELS:
        model_rows = display.loc[model]
        for i, setting in enumerate(TRAIN_SETTINGS):
            model_cell = rf"\multirow{{3}}{{*}}{{{escape_latex(model)}}}" if i == 0 else ""
            setting_cell = SETTING_LABELS[setting]
            vals = [model_rows.loc[setting, lookback] for lookback in LOOKBACKS]
            lines.append(" & ".join([model_cell, setting_cell] + vals) + r" \\")
        if model != MODELS[-1]:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{Mean {metric_display} across all countries and horizons for deep learning models. "
            r"Bold indicates the best look-back window within each row.}}",
            rf"\label{{tab:deep-window-summary-{metric_display.lower()}}}",
            r"\end{table}",
        ]
    )

    out_path = OUT_DIR / f"deep_window_summary_{metric_display}.tex"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    df = collect_records()
    long_path = OUT_DIR / "deep_window_metrics_long.csv"
    df.rename(columns={"WIS80": "IS80"}).to_csv(long_path, index=False)

    summary = (
        df.groupby(["method", "train_setting", "lookback"], dropna=False)[[metric_key for metric_key, _ in METRICS]]
        .mean()
        .reset_index()
    )
    summary_path = OUT_DIR / "deep_window_metrics_summary.csv"
    summary.rename(columns={"WIS80": "IS80"}).to_csv(summary_path, index=False)

    print(f"[OK] wrote {long_path} rows={len(df)}")
    print(f"[OK] wrote {summary_path} rows={len(summary)}")

    for metric_key, metric_display in METRICS:
        full_path = make_full_metric_table(df, metric_key, metric_display)
        summary_tex_path = make_summary_table(df, metric_key, metric_display)
        nonempty = int(df[metric_key].notna().sum())
        print(f"[OK] wrote {full_path} nonempty={nonempty}")
        print(f"[OK] wrote {summary_tex_path}")


if __name__ == "__main__":
    main()
