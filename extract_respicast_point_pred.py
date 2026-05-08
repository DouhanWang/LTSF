# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations
from pathlib import Path
import pandas as pd

COUNTRY_FULL = {
    "BE": "Belgium",
    "CZ": "Czechia",
    "DK": "Denmark",
    "FR": "France",
    "IE": "Ireland",
    "IT": "Italy",
    "NL": "Netherlands",
    "PL": "Poland",
    "RO": "Romania",
}

def build_respicast_wide_targetcol(
    respicast_dir: str | Path | None = None,
    out_root: str | Path | None = None,
    countries: list[str] | None = None,
    target_name: str = "ILI incidence",
    steps=(1, 2, 3, 4),
    clip_lower_at_zero: bool = True,
) -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir  

    if countries is None:
        countries = ["BE", "CZ", "DK", "FR", "IE", "IT", "NL", "PL", "RO"]

    if respicast_dir is None:
        respicast_dir = project_root / "dataset" / "Respicast"
    else:
        respicast_dir = Path(respicast_dir)

    if out_root is None:
        out_root = project_root / "results"
    else:
        out_root = Path(out_root)

    csv_files = sorted(list(respicast_dir.rglob("*.csv")) + list(respicast_dir.rglob("*.CSV")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files under: {respicast_dir.resolve()}")

    keep_cols = [
        "origin_date", "target", "target_end_date",
        "horizon", "location", "output_type", "output_type_id", "value"
    ]

    frames = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        miss = [c for c in keep_cols if c not in df.columns]
        if miss:
            print(f"[WARN] Skip {fp.name}: missing {miss}")
            continue
        frames.append(df[keep_cols].copy())

    if not frames:
        raise RuntimeError("No valid Respicast CSVs (missing required columns).")

    all_df = pd.concat(frames, ignore_index=True)

   
    all_df = all_df[all_df["target"] == target_name].copy()
    all_df["target_end_date"] = pd.to_datetime(all_df["target_end_date"], errors="coerce")
    all_df["horizon"] = pd.to_numeric(all_df["horizon"], errors="coerce").astype("Int64")
    all_df["value"] = pd.to_numeric(all_df["value"], errors="coerce")
    all_df["output_type"] = all_df["output_type"].astype(str).str.lower()
    all_df["output_type_id"] = pd.to_numeric(all_df["output_type_id"], errors="coerce")
    all_df = all_df.dropna(subset=["target_end_date", "horizon", "location", "output_type", "value"])

    for country in countries:
        full = COUNTRY_FULL.get(country, country)
        out_dir = out_root / f"Respicast_real_{full}_ILI"
        out_dir.mkdir(parents=True, exist_ok=True)

        df_c = all_df[all_df["location"] == country].copy()
        if df_c.empty:
            print(f"[INFO] No rows for country={country}")
            continue

        for step in steps:
            df_h = df_c[df_c["horizon"] == step].copy()
            if df_h.empty:
                print(f"[INFO] No rows for {country} horizon={step}")
                continue

            # median -> target
            med = df_h[df_h["output_type"] == "median"][["target_end_date", "value"]].copy()
            med = med.rename(columns={"target_end_date": "date", "value": "target"})

            # quantile 0.1/0.9 -> lower80/upper80
            q10 = df_h[(df_h["output_type"] == "quantile") & (df_h["output_type_id"].sub(0.1).abs() < 1e-9)][
                ["target_end_date", "value"]
            ].copy()
            q90 = df_h[(df_h["output_type"] == "quantile") & (df_h["output_type_id"].sub(0.9).abs() < 1e-9)][
                ["target_end_date", "value"]
            ].copy()

            q10 = q10.rename(columns={"target_end_date": "date", "value": "lower80"})
            q90 = q90.rename(columns={"target_end_date": "date", "value": "upper80"})

            wide = med.merge(q10, on="date", how="outer").merge(q90, on="date", how="outer")

          
            wide = wide.sort_values("date").groupby("date", as_index=False).last()

           
            wide = wide[["date", "target", "lower80", "upper80"]].copy()

            if clip_lower_at_zero:
                wide["lower80"] = wide["lower80"].clip(lower=0.0)

            wide["date"] = pd.to_datetime(wide["date"]).dt.strftime("%Y-%m-%d")

            out_path = out_dir / f"rolling_pred_step{step}.csv"
            wide.to_csv(out_path, index=False)
            print(f"[OK] Saved {out_path} (rows={len(wide)})")


if __name__ == "__main__":
    build_respicast_wide_targetcol()