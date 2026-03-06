import numpy as np
import pandas as pd
from pathlib import Path

IN_PATH = Path("results/metrics_tables/point_metrics_long_real_sim.csv")
OUT_DIR = Path("results/metrics_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 你要显示的 dataset_type：如果只要 real 就改成 "real"
DATASET_TYPE = "real"

COUNTRIES = ["Belgium","Czechia","Denmark","France","Ireland","Italy","Netherlands","Poland","Romania"]
STEPS = [1,2,3,4]
ALL_METRICS = ["MAE", "wMAPE", "WIS80_mean"]

# 列结构：方法 -> 子列(训练设置)
COL_PLAN = [
    ("Naive", [""],
     ),
    ("ARIMA", [""],
     ),
    ("DLinear", ["real","augmented","combined"]),
    ("LSTM", ["real","augmented","combined"]),
    ("Autoformer", ["real","augmented","combined"]),
    ("TabPFN_ts", [""],
     ),
    ("Respicast", [""],
     ),
    ("Ensemble", [""],
     ),
]

def fmt(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{x:.4g}"

def escape(s: str) -> str:
    return s.replace("_", "\\_")

def build_columns():
    """Return list of (method, train_setting) for table columns in order."""
    cols = []
    for m, subs in COL_PLAN:
        for sub in subs:
            cols.append((m, sub))
    return cols

def make_rows(metrics):
    """Return list of (country, step, metric) row keys in order."""
    rows = []
    for c in COUNTRIES:
        for s in STEPS:
            for met in metrics:
                rows.append((c, s, met))
    return rows

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH}")

    df_all = pd.read_csv(IN_PATH)
    df_all["step"] = df_all["step"].astype(int)
    df_all["train_setting"] = df_all["train_setting"].fillna("").astype(str)

    metric_display = {
        "MAE": "MAE",
        "wMAPE": "wMAPE",
        "WIS80_mean": "mean WIS",
    }

    for ONE_METRIC in ALL_METRICS:
        METRICS = [ONE_METRIC]

        # filter (IMPORTANT: use df_all, do NOT overwrite it)
        df_sub = df_all[
            (df_all["dataset_type"] == DATASET_TYPE) &
            (df_all["country"].isin(COUNTRIES)) &
            (df_all["step"].isin(STEPS)) &
            (df_all["metric"].isin(METRICS))
        ].copy()

        # pivot to wide
        wide = df_sub.pivot_table(
            index=["country", "step", "metric"],
            columns=["method", "train_setting"],
            values="value",
            aggfunc="first"
        )

        # enforce column order
        col_order = pd.MultiIndex.from_tuples(build_columns(), names=["method", "train_setting"])
        wide = wide.reindex(columns=col_order)

        # enforce row order (IMPORTANT: only current METRICS)
        row_order = pd.MultiIndex.from_tuples(make_rows(METRICS), names=["country", "step", "metric"])
        wide = wide.reindex(row_order)

        # format as strings + bold row best
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

        # ---------- LaTeX rendering ----------
        n_data_cols = wide_str.shape[1]
        colspec = "lll" + "c" * n_data_cols

        header1 = ["", "", ""]
        header2 = ["Country", "Step", "Metric"]
        for m, subs in COL_PLAN:
            header1.append(f"\\multicolumn{{{len(subs)}}}{{c}}{{{escape(m)}}}")
            for sub in subs:
                header2.append(escape(sub) if sub != "" else "")

        h1 = " & ".join(header1) + " \\\\"
        h2 = " & ".join(header2) + " \\\\"

        lines = []
        lines.append(r"\begin{landscape}")
        lines.append(r"\begin{table}[t]")
        lines.append("\\vspace*{-0.6cm}")
        lines.append("\\centering")
        lines.append("\\hspace*{-0.6cm}")
        lines.append("\\scriptsize")
        lines.append("\\setlength\\tabcolsep{1.8pt}")
        lines.append("\\renewcommand{\\arraystretch}{1.0}")
        lines.append("\\resizebox{\\linewidth}{!}{%")
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append("\\toprule")
        lines.append(h1)
        lines.append(h2)
        lines.append("\\midrule")

        rows = wide_str.index.tolist()
        data = wide_str.to_numpy()

        country_span = len(STEPS) * len(METRICS)  # here METRICS=1 -> span=4
        step_span = len(METRICS)                  # -> 1

        for i, (country, step, metric) in enumerate(rows):
            if i % country_span == 0:
                c_cell = f"\\multirow{{{country_span}}}{{*}}{{{escape(country)}}}"
            else:
                c_cell = ""

            if i % step_span == 0:
                s_cell = f"\\multirow{{{step_span}}}{{*}}{{step{step}}}"
            else:
                s_cell = ""

            m_cell = escape(metric_display.get(metric, metric))

            vals = list(data[i, :])
            lines.append(" & ".join([c_cell, s_cell, m_cell] + vals) + " \\\\")

            if (i + 1) % country_span == 0 and (i + 1) != len(rows):
                lines.append("\\midrule")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}%")
        lines.append("}")
        lines.append("\\vspace*{-0.2cm}")
        lines.append(f"\\caption{{{escape(DATASET_TYPE)} dataset: one-table summary across 9 countries, 4 horizons, and metric {escape(ONE_METRIC)}.}}")
        lines.append(f"\\label{{tab:big_{escape(DATASET_TYPE)}_{escape(ONE_METRIC)}}}")
        lines.append(r"\end{table}")
        lines.append(r"\end{landscape}")

        out_tex = OUT_DIR / f"big_table_{DATASET_TYPE}_{ONE_METRIC}.tex"
        out_tex.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] wrote {out_tex} (rows={len(rows)}, nonempty={wide.notna().sum().sum()})")

if __name__ == "__main__":
    main()