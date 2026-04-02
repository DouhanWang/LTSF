import numpy as np
import pandas as pd
from pathlib import Path

IN_PATH = Path("results/metrics_tables/point_metrics_long_real_sim.csv")
OUT_DIR = Path("results/metrics_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_TYPE = "real"

COUNTRIES = ["Belgium","Czechia","Denmark","France","Ireland","Italy","Netherlands","Poland","Romania"]
STEPS = [1, 2, 3, 4]
ALL_METRICS = ["MAE", "wMAPE", "WIS80_mean"]

# Flat column order: 确保与 all_metrics.py 抓取的名字绝对一致
COL_PLAN = [
    ("Naive",      ""),
    ("ARIMA",      ""),
    ("DLinear",    "real"),
    ("DLinear",    "augmented"),
    ("DLinear",    "combined"),
    ("LSTM",       "real"),
    ("LSTM",       "augmented"),
    ("LSTM",       "combined"),
    ("Autoformer", "real"),
    ("Autoformer", "augmented"),
    ("Autoformer", "combined"),
    ("TabPFN_ts",  ""),
    ("Respicast",  ""),
    ("Ensemble",   ""),
    ("Ensembleun", ""),  # 新增的 Unweighted Ensemble
]

# Header groups for row 1:
# (display_name, n_cols, use_multirow2, sub_labels_row2)
HEADER_GROUPS = [
    ("Naive ARIMA", 2, False, ["real", "real"]),
    ("DLinear",     3, False, ["real", "aug", "combined"]),
    ("LSTM",        3, False, ["real", "aug", "combined"]),
    ("Autoformer",  3, False, ["real", "aug", "combined"]),
    ("TabPFN\\_ts", 1, True,  [""]),
    ("Respicast",   1, True,  [""]),
    ("Ensemble",    1, True,  [""]),
    ("Ensemble\\_un",1, True,  [""]),  # 新增，表头显示带下划线
]

# colspec: ll (Country, Step) + one group per HEADER_GROUPS entry
# Result: ll cc ccc ccc ccc c c c c (17 cols total)
_group_specs = ["".join("c" for _ in range(n)) for _, n, _, _ in HEADER_GROUPS]
COLSPEC = "ll " + " ".join(_group_specs)

# cmidrule positions (1-indexed; cols 1-2 are label cols)
def build_cmidrule():
    parts = []
    col = 3  # first data col
    for _, n, use_multirow, _ in HEADER_GROUPS:
        if not use_multirow and n > 1:
            parts.append(f"\\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    return " ".join(parts)

CMIDRULE = build_cmidrule()

def fmt(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{x:.4g}"

def escape(s: str) -> str:
    return s.replace("_", "\\_")

def build_header():
    """Return (h1_line, cmidrule_line, h2_line) as LaTeX strings."""
    h1 = ["\\multirow{2}{*}{\\textbf{Country}}", "\\multirow{2}{*}{\\textbf{Step}}"]
    h2 = ["", ""]

    for display, n, use_multirow, sub_labels in HEADER_GROUPS:
        if use_multirow:
            h1.append(f"\\multirow{{2}}{{*}}{{\\textbf{{{display}}}}}")
        else:
            h1.append(f"\\multicolumn{{{n}}}{{c}}{{\\textbf{{{display}}}}}")
        for lbl in sub_labels:
            h2.append(lbl)

    return (
        " & ".join(h1) + " \\\\",
        CMIDRULE,
        " & ".join(h2) + " \\\\",
    )

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH}")

    df_all = pd.read_csv(IN_PATH)
    df_all["step"] = df_all["step"].astype(int)
    df_all["train_setting"] = df_all["train_setting"].fillna("").astype(str)

    col_index = pd.MultiIndex.from_tuples(COL_PLAN, names=["method", "train_setting"])
    row_order = pd.MultiIndex.from_tuples(
        [(c, s) for c in COUNTRIES for s in STEPS],
        names=["country", "step"],
    )

    metric_display = {
        "MAE":       "MAE",
        "wMAPE":     "wMAPE",
        "WIS80_mean": "mean WIS",
    }

    h1_line, cmidrule_line, h2_line = build_header()

    for ONE_METRIC in ALL_METRICS:
        df_sub = df_all[
            (df_all["dataset_type"] == DATASET_TYPE) &
            (df_all["country"].isin(COUNTRIES)) &
            (df_all["step"].isin(STEPS)) &
            (df_all["metric"] == ONE_METRIC)
        ].copy()

        wide = df_sub.pivot_table(
            index=["country", "step"],
            columns=["method", "train_setting"],
            values="value",
            aggfunc="first",
        )
        wide = wide.reindex(columns=col_index)
        wide = wide.reindex(row_order)

        # Bold minimum value in each row
        row_best = wide.min(axis=1, skipna=True)
        wide_str = wide.astype(object)
        for idx in wide.index:
            best = row_best.loc[idx]
            for col in wide.columns:
                v = wide.loc[idx, col]
                if pd.isna(v):
                    wide_str.loc[idx, col] = ""
                else:
                    s = fmt(float(v))
                    if pd.notna(best) and np.isclose(float(v), float(best)):
                        wide_str.loc[idx, col] = f"\\textbf{{{s}}}"
                    else:
                        wide_str.loc[idx, col] = s

        # Build LaTeX lines
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\makebox[\textwidth][c]{")
        lines.append(r"\fontsize{7.2pt}{10pt}\selectfont")
        lines.append(r"\addtolength{\tabcolsep}{-5pt}")
        lines.append(f"\\begin{{tabular}}{{{COLSPEC}}}")
        lines.append(r"\toprule")
        lines.append(h1_line)
        lines.append(cmidrule_line)
        lines.append(h2_line)
        lines.append(r"\midrule")

        data = wide_str.to_numpy()
        country_span = len(STEPS)

        for i, (country, step) in enumerate(wide_str.index.tolist()):
            c_cell = f"\\multirow{{{country_span}}}{{*}}{{{escape(country)}}}" if i % country_span == 0 else ""
            s_cell = f"step{step}"
            vals = list(data[i, :])
            lines.append(" & ".join([c_cell, s_cell] + vals) + " \\\\")
            # midrule between countries, but not after the last
            if (i + 1) % country_span == 0 and (i + 1) != len(wide_str):
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")

        disp = metric_display.get(ONE_METRIC, ONE_METRIC)
        label_metric = ONE_METRIC.replace("_", "-")
        lines.append(
            f"\\caption{{{escape(DATASET_TYPE)} dataset: one-table summary "
            f"across 9 countries, 4 horizons, and metric {disp}.}}"
        )
        lines.append(f"\\label{{tab:big-{DATASET_TYPE}-{label_metric}}}")
        lines.append(r"\end{table}")

        out_tex = OUT_DIR / f"big_table_{DATASET_TYPE}_{ONE_METRIC}.tex"
        out_tex.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] wrote {out_tex}  (rows={len(wide_str)}, nonempty={wide.notna().sum().sum()})")

if __name__ == "__main__":
    main()