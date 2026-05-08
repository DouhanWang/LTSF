# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare auto-ARIMA and ARIMAX on one country's ILI season holdout."
    )
    parser.add_argument("--data", default="dataset/real_Italy_ILI.csv")
    parser.add_argument("--target", default="incidenza")
    parser.add_argument("--season-col", default="season_id")
    parser.add_argument("--week-col", default="settimana")
    parser.add_argument("--holdout-season", type=int, default=None)
    parser.add_argument("--mode", choices=["holdout", "rolling4"], default="rolling4")
    parser.add_argument("--known-weeks", type=int, default=4)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument("--max-p", type=int, default=5)
    parser.add_argument("--max-d", type=int, default=2)
    parser.add_argument("--max-q", type=int, default=5)
    parser.add_argument("--max-order", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def add_features(df, season_col):
    df = df.copy()
    season_len = df.groupby(season_col)[season_col].transform("size")
    df["season_week"] = df.groupby(season_col).cumcount() + 1
    angle = 2.0 * math.pi * (df["season_week"] - 1) / season_len
    df["season_week_sin"] = np.sin(angle)
    df["season_week_cos"] = np.cos(angle)
    df["season_week_scaled"] = (df["season_week"] - 1) / (season_len - 1).replace(0, np.nan)
    df["season_week_scaled"] = df["season_week_scaled"].fillna(0.0)
    df["season_index"] = df[season_col].astype(float)
    return df


def exog_matrix(df):
    cols = ["season_week_sin", "season_week_cos", "season_week_scaled", "season_index"]
    return df[cols].astype(float).to_numpy()


def scale_exog(train_exog, future_exog):
    mu = train_exog.mean(axis=0)
    sd = train_exog.std(axis=0)
    sd[sd == 0] = 1.0
    return (train_exog - mu) / sd, (future_exog - mu) / sd


def candidate_orders(max_p, max_d, max_q, max_order):
    for d in range(max_d + 1):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                if max_order is not None and p + q > max_order:
                    continue
                yield (p, d, q)


def fit_best_model(y, exog, args):
    best = None
    best_score = np.inf

    for order in candidate_orders(args.max_p, args.max_d, args.max_q, args.max_order):
        if len(y) < max(10, sum(order) + 5):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                model = SARIMAX(
                    y,
                    exog=exog,
                    order=order,
                    seasonal_order=(0, 0, 0, 0),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False, maxiter=args.maxiter)
            if np.isfinite(res.aic) and res.aic < best_score:
                best = (order, res)
                best_score = float(res.aic)
        except Exception:
            continue

    if best is None:
        raise RuntimeError("No ARIMA/SARIMAX candidate could be fitted.")
    return best


def forecast_once(train_df, future_df, args, use_exog):
    y = train_df[args.target].astype(float).to_numpy()

    if use_exog:
        train_exog = exog_matrix(train_df)
        future_exog = exog_matrix(future_df)
        train_exog, future_exog = scale_exog(train_exog, future_exog)
    else:
        train_exog = None
        future_exog = None

    order, res = fit_best_model(y, train_exog, args)
    forecast = res.get_forecast(steps=len(future_df), exog=future_exog).predicted_mean
    return order, np.asarray(forecast, dtype=float)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mask = y_true != 0
    mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100.0) if mask.any() else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def evaluate_holdout(df, args):
    holdout = args.holdout_season
    train_df = df[df[args.season_col] != holdout]
    test_df = df[df[args.season_col] == holdout]
    y_true = test_df[args.target].to_numpy(dtype=float)

    rows = []
    for name, use_exog in [("auto_arima", False), ("arimax", True)]:
        order, y_pred = forecast_once(train_df, test_df, args, use_exog)
        row = {"model": name, "order": order, "n_pred": len(y_true)}
        row.update(metrics(y_true, y_pred))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_rolling4(df, args):
    holdout = args.holdout_season
    base_train = df[df[args.season_col] != holdout]
    test_df = df[df[args.season_col] == holdout].reset_index(drop=True)

    records = []
    max_origin = len(test_df) - args.known_weeks - args.pred_len + 1

    for origin in range(max_origin):
        known = args.known_weeks + origin
        train_df = pd.concat([base_train, test_df.iloc[:known]], ignore_index=True)
        future_df = test_df.iloc[known : known + args.pred_len]
        y_true = future_df[args.target].to_numpy(dtype=float)

        if not args.quiet:
            print(f"origin {origin + 1}/{max_origin}: known={known}")

        for name, use_exog in [("auto_arima", False), ("arimax", True)]:
            order, y_pred = forecast_once(train_df, future_df, args, use_exog)
            for horizon, (actual, pred) in enumerate(zip(y_true, y_pred), start=1):
                records.append(
                    {
                        "model": name,
                        "origin": origin,
                        "horizon": horizon,
                        "order": order,
                        "actual": actual,
                        "pred": float(pred),
                    }
                )

    pred_df = pd.DataFrame(records)
    rows = []
    for name, g in pred_df.groupby("model"):
        row = {"model": name, "n_pred": len(g)}
        row.update(metrics(g["actual"], g["pred"]))
        rows.append(row)
    summary = pd.DataFrame(rows)

    horizon_rows = []
    for (name, horizon), g in pred_df.groupby(["model", "horizon"]):
        row = {"model": name, "horizon": horizon, "n_pred": len(g)}
        row.update(metrics(g["actual"], g["pred"]))
        horizon_rows.append(row)
    by_horizon = pd.DataFrame(horizon_rows)
    return summary, by_horizon, pred_df


def main():
    args = parse_args()
    path = Path(args.data)
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df = add_features(df, args.season_col)

    if args.holdout_season is None:
        args.holdout_season = int(df[args.season_col].max())

    print(f"data={path}")
    print(f"holdout_season={args.holdout_season}")
    print(
        "search="
        f"p=0..{args.max_p}, d=0..{args.max_d}, q=0..{args.max_q}, "
        f"max_order={args.max_order}, seasonal=False"
    )

    if args.mode == "holdout":
        summary = evaluate_holdout(df, args)
        print("\nsummary")
        print(summary.to_string(index=False))
        return

    summary, by_horizon, _ = evaluate_rolling4(df, args)
    print("\nsummary")
    print(summary.to_string(index=False))
    print("\nby_horizon")
    print(by_horizon.sort_values(["horizon", "model"]).to_string(index=False))


if __name__ == "__main__":
    main()
