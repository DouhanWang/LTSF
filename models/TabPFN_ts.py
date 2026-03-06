import argparse
import os
import numpy as np
import pandas as pd
from epi4cast.utils.tools import visual
from tabpfn_time_series import (
    FeatureTransformer,
    TabPFNTimeSeriesPredictor,
    TabPFNMode,
    TimeSeriesDataFrame,
)
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
)

def _maybe_build_timestamp_from_year_week(df: pd.DataFrame) -> pd.DataFrame:
    # 如果已经有 timestamp/date，就不动
    if "timestamp" in df.columns or "date" in df.columns:
        return df

    # 如果有 anno + settimana，就自动生成 timestamp（周一）
    if "anno" in df.columns and "settimana" in df.columns:
        out = df.copy()
        out["timestamp"] = pd.to_datetime(
            out["anno"].astype(str) + "-W" + out["settimana"].astype(str).str.zfill(2) + "-1",
            format="%G-W%V-%u",
            errors="raise",
        )
        return out

    return df
def _get_time_col(df: pd.DataFrame) -> str:
    if "timestamp" in df.columns:
        return "timestamp"
    if "date" in df.columns:
        return "date"
    raise ValueError("CSV 里需要有 'timestamp' 或 'date' 作为时间列。")


def _pad_future_rows(curr_test_df: pd.DataFrame, need_len: int, time_col: str, item_col: str, target_col: str):
    """补齐未来时间戳，默认 freq=7D（周频）。"""
    if len(curr_test_df) >= need_len:
        return curr_test_df

    missing = need_len - len(curr_test_df)
    if curr_test_df.empty:
        raise ValueError("curr_test_df 为空，无法补齐未来时间戳。请检查 train_end/test_size。")

    last_t = pd.to_datetime(curr_test_df[time_col].iloc[-1])
    future_times = [last_t + pd.Timedelta(days=7 * (j + 1)) for j in range(missing)]

    future_df = pd.DataFrame({
        item_col: [curr_test_df[item_col].iloc[0]] * missing,
        time_col: future_times,
        target_col: [np.nan] * missing,
    })

    out = pd.concat([curr_test_df, future_df], ignore_index=True)
    return out

def WIS_interval(pred_lower, pred_upper, true, alpha: float):
    """
    pred_lower, pred_upper, true: numpy arrays same shape (T,)
    alpha: e.g. 0.2 for 80% interval
    """
    pred_lower = np.asarray(pred_lower, dtype=float)
    pred_upper = np.asarray(pred_upper, dtype=float)
    true = np.asarray(true, dtype=float)

    width = pred_upper - pred_lower
    below = (true < pred_lower) * (pred_lower - true)
    above = (true > pred_upper) * (true - pred_upper)
    wis = width + (2.0 / alpha) * below + (2.0 / alpha) * above
    return wis
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="CSV path")
    parser.add_argument("--out_dir", type=str, default="results/tabpfn_ts_rolling", help="output folder")
    parser.add_argument("--item_id", type=int, default=None, help="只跑某一个 item_id（可选）")
    parser.add_argument("--item_col", type=str, default="item_id", help="item id column name")
    parser.add_argument("--target", type=str, default="target", help="target column name")
    parser.add_argument("--test_size", type=int, default=21, help="rolling 测试点数量（每次 origin +1）")
    parser.add_argument("--pred_len", type=int, default=4, help="预测步数（默认 4）")
    parser.add_argument("--freq_days", type=int, default=7, help="补未来时间戳的步长（天），周频=7")
    parser.add_argument("--tabpfn_mode", type=str, default="client", choices=["client", "local"], help="TabPFNMode")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_raw = pd.read_csv(args.data_path)
    df_raw = _maybe_build_timestamp_from_year_week(df_raw)
    if args.target != "target":
        if args.target not in df_raw.columns:
            raise ValueError(f"Target column '{args.target}' not in CSV columns: {list(df_raw.columns)}")
        df_raw = df_raw.rename(columns={args.target: "target"})
        args.target = "target"
    if args.item_col not in df_raw.columns:
        raise ValueError(f"找不到 item_col='{args.item_col}'，请检查 CSV 列名。")

    time_col = _get_time_col(df_raw)
    df_raw = df_raw.copy()
    df_raw[time_col] = pd.to_datetime(df_raw[time_col])

    # 可选：只跑一个 item_id
    if args.item_id is not None:
        df_raw = df_raw[df_raw[args.item_col] == args.item_id].copy()
        df_raw = df_raw.sort_values(time_col).reset_index(drop=True)

    # 必须保证 target 存在
    if args.target not in df_raw.columns:
        raise ValueError(f"找不到 target='{args.target}' 列。")

    total_len = len(df_raw)
    test_size = int(args.test_size)
    pred_len = int(args.pred_len)

    if total_len <= test_size:
        raise ValueError(f"数据太短 total_len={total_len} <= test_size={test_size}，无法 rolling。")

    train_end_index = total_len - test_size

    tabpfn_mode = TabPFNMode.CLIENT if args.tabpfn_mode.lower() == "client" else TabPFNMode.LOCAL
    predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=tabpfn_mode)

    selected_features = [RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]
    feature_transformer = FeatureTransformer(selected_features)

    print(f"[TabPFN_ts Rolling] total_len={total_len} test_size={test_size} pred_len={pred_len}")
    print(f"Train end starts at index {train_end_index}")
    print(f"Time column: {time_col} | Item column: {args.item_col} | Target: {args.target}")

    # 保存：按 step(1..pred_len) 分开存
    all_predictions_by_step = {step: [] for step in range(1, pred_len + 1)}

    for step in range(1, pred_len + 1):
        print(f"\n========== Running {step}-step (save only step {step}) ==========")

        for i in range(0, test_size):
            current_train_end = train_end_index + i

            curr_train_df = df_raw.iloc[:current_train_end].copy()

            desired_end = current_train_end + pred_len
            real_end = min(desired_end, total_len)
            curr_test_df = df_raw.iloc[current_train_end:real_end].copy()

            if curr_test_df.empty:
                break

            # 不够 pred_len 就补未来时间戳（只为能算 step3/4）
            if len(curr_test_df) < pred_len:
                # 用 args.freq_days 而不是写死 7
                last_t = pd.to_datetime(curr_test_df[time_col].iloc[-1])
                missing = pred_len - len(curr_test_df)
                future_times = [last_t + pd.Timedelta(days=args.freq_days * (j + 1)) for j in range(missing)]
                future_df = pd.DataFrame({
                    args.item_col: [curr_test_df[args.item_col].iloc[0]] * missing,
                    time_col: future_times,
                    args.target: [np.nan] * missing,
                })
                curr_test_df = pd.concat([curr_test_df, future_df], ignore_index=True)

            if len(curr_test_df) < step:
                continue

            # TabPFN_ts：test 的 target 必须 mask 成 NaN
            curr_test_df[args.target] = np.nan

            # TSDF
            train_tsdf = TimeSeriesDataFrame(curr_train_df)
            test_tsdf = TimeSeriesDataFrame(curr_test_df)

            # Feature transform
            train_tsdf_feat, test_tsdf_feat = feature_transformer.transform(train_tsdf, test_tsdf)

            # Predict
            pred_chunk = predictor.predict(train_tsdf_feat, test_tsdf_feat)
            pred_chunk_df = pred_chunk.reset_index(drop=False)

            # save only the step-th row
            step_df = pred_chunk_df.iloc[[step - 1]].copy()
            step_df["saved_step"] = step
            step_df["origin_i"] = i
            step_df["train_end"] = current_train_end
            all_predictions_by_step[step].append(step_df)

            print(f"[step={step}] origin i={i}, train_end={current_train_end}, "
                  f"pred_rows={len(pred_chunk_df)}, saved_row={step}")

    # merge & save
    pred_steps = {}  # step -> DataFrame
    for step in range(1, pred_len + 1):
        if not all_predictions_by_step[step]:
            print(f"[Warn] step={step} has no predictions.")
            continue

        pred_step = pd.concat(all_predictions_by_step[step], ignore_index=True)
        pred_steps[step] = pred_step  # <-- NEW: keep in memory

        out_csv = os.path.join(args.out_dir, f"tabpfn_ts_pred_step{step}.csv")
        pred_step.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}  (rows={len(pred_step)})")

    # ==========================
    # WIS80 per-step (alpha=0.2)
    # ==========================
    alpha = 0.2
    true_s = df_raw["target"].to_numpy(dtype=float)

    def _qcol(df: pd.DataFrame, q):
        # handle numeric or string column names
        if q in df.columns:
            return df[q]
        qs = str(q)
        if qs in df.columns:
            return df[qs]
        raise KeyError(f"Quantile column {q} not found. Columns={df.columns.tolist()}")

    def compute_wis_for_step(pred_step_df: pd.DataFrame, step: int) -> np.ndarray:
        # need quantiles + train_end
        lower = _qcol(pred_step_df, 0.1).to_numpy(dtype=float)
        upper = _qcol(pred_step_df, 0.9).to_numpy(dtype=float)
        train_end = pred_step_df["train_end"].to_numpy(dtype=int)

        truth_idx = train_end + (step - 1)
        mask = (truth_idx >= 0) & (truth_idx < len(true_s)) & np.isfinite(lower) & np.isfinite(upper)

        lower = lower[mask]
        upper = upper[mask]
        truth = true_s[truth_idx[mask]]

        return WIS_interval(lower, upper, truth, alpha=alpha)

    for step in range(1, pred_len + 1):
        if step not in pred_steps:
            continue
        wis80 = compute_wis_for_step(pred_steps[step], step=step)
        npy_path = os.path.join(args.out_dir, f"wis80_point_step{step}.npy")
        np.save(npy_path, wis80)
        print(f"Saved: {npy_path}  (n={wis80.size}, mean={float(np.nanmean(wis80)) if wis80.size else None})")
    # ==========================
    # Plot step1..4 to: test_result/<data>_TabPFN_ts/rolling_test_step_k.png
    # ==========================
    data_tag = os.path.splitext(os.path.basename(args.data_path))[0]  # e.g. real_Italy_ILI
    model_tag = "TabPFN_ts"
    setting = f"{model_tag}_{data_tag}"  # e.g. TabPFN_ts_real_Italy_ILI

    # infer epi4cast root from out_dir
    out_dir_abs = os.path.abspath(args.out_dir)
    # expected: .../epi4cast/results/<setting>  -> root is parent of "results"
    parent_name = os.path.basename(os.path.dirname(out_dir_abs))
    if parent_name.lower() == "results":
        epi_root = os.path.dirname(os.path.dirname(out_dir_abs))
    else:
        # fallback: put test_results next to out_dir's parent
        epi_root = os.path.dirname(out_dir_abs)

    plot_dir = os.path.join(epi_root, "test_results", setting)
    os.makedirs(plot_dir, exist_ok=True)

    true_series = pd.Series(
        df_raw["target"].to_numpy(dtype=float),
        index=pd.to_datetime(df_raw[time_col]),
        name="target",
    )

    def _get_pred_time_col(dfp: pd.DataFrame) -> str:
        if time_col in dfp.columns:
            return time_col
        if "timestamp" in dfp.columns:
            return "timestamp"
        if "date" in dfp.columns:
            return "date"
        raise KeyError(f"No time column found in prediction df. Columns={dfp.columns.tolist()}")

    def _get_q(dfp: pd.DataFrame, q):
        if q in dfp.columns:
            return dfp[q]
        qs = str(q)
        if qs in dfp.columns:
            return dfp[qs]
        return None

    def _get_center(dfp: pd.DataFrame) -> pd.Series:
        if 0.5 in dfp.columns:
            return dfp[0.5]
        if "0.5" in dfp.columns:
            return dfp["0.5"]
        lo = _get_q(dfp, 0.1)
        hi = _get_q(dfp, 0.9)
        if lo is not None and hi is not None:
            return (lo + hi) / 2.0
        raise KeyError(f"Cannot find point forecast (0.5) or interval (0.1/0.9). Columns={dfp.columns.tolist()}")

    for step in [1, 2, 3, 4]:
        if step not in pred_steps:
            print(f"[Warn] No pred_steps[{step}] to plot.")
            continue

        dfp = pred_steps[step].copy()
        tcol = _get_pred_time_col(dfp)
        idx = pd.to_datetime(dfp[tcol])

        center = _get_center(dfp).to_numpy(dtype=float)
        lo = _get_q(dfp, 0.1)
        hi = _get_q(dfp, 0.9)

        preds_series = pd.Series(center, index=idx, name=f"pred_step{step}")

        lower_series = None
        upper_series = None
        if lo is not None and hi is not None:
            lower_series = pd.Series(lo.to_numpy(dtype=float), index=idx, name=f"lower_step{step}")
            upper_series = pd.Series(hi.to_numpy(dtype=float), index=idx, name=f"upper_step{step}")

        fig_path = os.path.join(plot_dir, f"rolling_test_step_{step}.png")

        visual(
            true=true_series,
            preds=preds_series,
            path=fig_path,
            lower=lower_series,
            upper=upper_series,
            seq_len=train_end_index,
        )
        print(f"Saved plot: {fig_path}")
    print("\nDone.")
    print("Saved counts:", {k: len(v) for k, v in all_predictions_by_step.items()})


if __name__ == "__main__":
    main()
