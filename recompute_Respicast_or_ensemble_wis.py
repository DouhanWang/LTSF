import os
import numpy as np
import pandas as pd

from utils.metrics import mean_WIS_interval  # 你原来的函数


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns found: {candidates}. Available: {list(df.columns)}")


def load_pred_csv(path):
    df = pd.read_csv(path)
    date_col = pick_col(df, ["date", "Date", "timestamp", "ds", "time"])
    df = df.rename(columns={date_col: "date"})
    return df


def wis_pointwise(pred_lower, pred_upper, true, alpha):
    """
    Return per-timepoint WIS (interval score) array, same length as inputs.
    """
    pred_lower = np.asarray(pred_lower, dtype=float)
    pred_upper = np.asarray(pred_upper, dtype=float)
    true = np.asarray(true, dtype=float)

    width = pred_upper - pred_lower
    below = (true < pred_lower) * (pred_lower - true)
    above = (true > pred_upper) * (true - pred_upper)
    is_alpha = width + (2.0 / alpha) * below + (2.0 / alpha) * above
    return is_alpha


def compute_and_save(country, step, alpha=0.2):
    ens_dir = f"results/Ensemble_real_{country}_ILI"
    ens_path = f"{ens_dir}/rolling_pred_step{step}.csv"
    naive_path = f"results/Naive_real_{country}_ILI/rolling_pred_step{step}.csv"

    ens = load_pred_csv(ens_path)
    nai = load_pred_csv(naive_path)

    # Respicast columns
    lower_col = pick_col(ens, ["lower80", "lower", "lo80", "q0.1", "q10", "lower_80"])
    upper_col = pick_col(ens, ["upper80", "upper", "hi80", "q0.9", "q90", "upper_80"])
    pred_col  = pick_col(ens, ["pred", "yhat", "forecast", "median", "mean"])

    # naive true column（按你实际列名扩展）
    true_col = pick_col(nai, ["true", "truth", "y", "target", "label", "gt", "obs", "value", "ILI"])

    lower = ens[lower_col].to_numpy(dtype=float)
    upper = ens[upper_col].to_numpy(dtype=float)

    true_all = nai[true_col].to_numpy(dtype=float)

    n_ens = len(lower)  # Respicast 的长度（upper 应该同长）
    if len(upper) != n_ens:
        raise ValueError(f"Respicast lower/upper length mismatch: {len(lower)} vs {len(upper)}")

    if len(true_all) < n_ens:
        raise ValueError(f"Naive true shorter than Respicast: {len(true_all)} < {n_ens}")

    # 从 Naive 的末尾取 n_ens 个（后对齐）
    true = true_all[-n_ens:]

    # 现在长度一致了，再做 mask
    mask = np.isfinite(true) & np.isfinite(lower) & np.isfinite(upper)

    true = true[mask]
    lower = lower[mask]
    upper = upper[mask]

    if true.size == 0:
        raise ValueError(f"After filtering non-NaN true, no points left for {country} step{step}.")

    if np.any(upper < lower):
        idx = np.where(upper < lower)[0][:5]
        raise ValueError(f"Found upper < lower in {country} step{step} at filtered rows {idx}.")



    # pointwise WIS (interval score)
    wis_points = wis_pointwise(lower, upper, true, alpha=alpha)

    # save
    os.makedirs(ens_dir, exist_ok=True)
    out_path = f"{ens_dir}/wis80_point_step{step}.npy"
    np.save(out_path, wis_points)

    # diagnostics
    mean_wis = float(np.mean(wis_points))
    picp = float(np.mean((true >= lower) & (true <= upper)))
    mean_width = float(np.mean(upper - lower))

    return out_path, mean_wis, picp, mean_width, len(wis_points)


def main():
    countries = ["Belgium", "Czechia", "Denmark", "France","Ireland","Italy","Netherlands","Poland","Romania"]  # TODO: 换成你的国家列表
    alpha = 0.2  # 80% interval

    for c in countries:
        print(f"\n=== {c} ===")
        for step in [1, 2, 3, 4]:
            out_path, mean_wis, picp, width, n = compute_and_save(c, step, alpha=alpha)
            print(f"step{step}: saved {out_path} | meanWIS80={mean_wis:.4f} | PICP80={picp:.3f} | width={width:.3f} | n={n}")


if __name__ == "__main__":
    main()