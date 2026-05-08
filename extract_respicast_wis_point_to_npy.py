# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import os
import numpy as np
import pandas as pd

IN_PATH = "dataset/latest-forecast_scores.csv"
OUT_ROOT = "results"

START_ORIGIN = "2024-11-17"
END_ORIGIN = "2025-04-09"


COUNTRY_MAP = {
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

TEAM_ID = "respicast"
MODEL_KEYWORD = "hubensemble"
TARGET_KEYWORD = "ILI"

INTERVAL_METRIC = "WIS"
POINT_METRIC_PRIORITY = ["AE", "MAE", "MSE", "RMSE"]


def parse_origin_date(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    dt2 = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return dt.fillna(dt2)


def pick_point_metric(df: pd.DataFrame) -> str | None:
    available = set(df["metric"].astype(str).str.strip().str.upper().unique())
    for m in POINT_METRIC_PRIORITY:
        if m.upper() in available:
            return m.upper()
    return None


def safe_name(name: str) -> str:

    return name.strip().replace(" ", "_")


def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Missing input: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    for c in ["team_id", "model_id", "metric", "target", "location"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "origin_date" not in df.columns:
        raise ValueError("Column 'origin_date' not found in input CSV.")

    df["origin_date_dt"] = parse_origin_date(df["origin_date"])
    if df["origin_date_dt"].isna().all():
        raise ValueError("Failed to parse 'origin_date' into datetime.")

    start_dt = pd.to_datetime(START_ORIGIN)
    end_dt = pd.to_datetime(END_ORIGIN)

    df = df[(df["origin_date_dt"] >= start_dt) & (df["origin_date_dt"] <= end_dt)].copy()

    df = df[
        (df["team_id"].str.lower() == TEAM_ID.lower())
        & (df["model_id"].str.lower().str.contains(MODEL_KEYWORD, na=False))
        & (df["target"].str.contains(TARGET_KEYWORD, case=False, na=False))
    ].copy()

    if df.empty:
        raise ValueError("No rows after filtering. Check TEAM/MODEL/TARGET/time window.")

    point_metric = pick_point_metric(df)

    sort_keys = [k for k in ["origin_date_dt", "target_end_date", "location", "horizon"] if k in df.columns]

    for code, full_name in COUNTRY_MAP.items():
        df_c = df[df["location"] == code].copy()
        if df_c.empty:
            print(f"[WARN] No rows for country={code} ({full_name}) after filters.")
            continue

        folder_country = safe_name(full_name)
        out_dir = os.path.join(OUT_ROOT, f"Respicast_real_{folder_country}_ILI")
        os.makedirs(out_dir, exist_ok=True)

        for step in [1, 2, 3, 4]:
            df_s = df_c[df_c["horizon"].astype(int) == step].copy()
            if df_s.empty:
                print(f"[WARN] {full_name} step{step}: no rows.")
                continue

            df_wis = df_s[df_s["metric"].str.upper() == INTERVAL_METRIC.upper()].copy()
            if df_wis.empty:
                print(f"[WARN] {full_name} step{step}: no WIS rows.")
                continue

            if sort_keys:
                df_wis = df_wis.sort_values(sort_keys)

            wis80 = df_wis["value_absolute"].to_numpy(dtype=float)

            point = None
            if point_metric is not None:
                df_p = df_s[df_s["metric"].str.upper() == point_metric.upper()].copy()
                if not df_p.empty:
                    if sort_keys:
                        df_p = df_p.sort_values(sort_keys)
                    point = df_p["value_absolute"].to_numpy(dtype=float)

            payload = {
                "country_code": code,
                "country": full_name,
                "step": step,
                "origin_date": df_wis["origin_date"].to_numpy(),
                "target_end_date": df_wis["target_end_date"].to_numpy() if "target_end_date" in df_wis.columns else None,
                "target": df_wis["target"].to_numpy(),
                "wis80": wis80,
                "point_metric": point_metric,
                "point": point,
            }

            out_path = os.path.join(out_dir, f"wis80_point_step{step}.npy")
            np.save(out_path, payload, allow_pickle=True)
            print(f"[OK] saved {out_path} | n={len(wis80)} | point_metric={point_metric}")

    print("Done.")


if __name__ == "__main__":
    main()