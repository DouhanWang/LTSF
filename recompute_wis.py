import os

import numpy as np
import pandas as pd


TARGET_LENGTHS = {1: 21, 2: 20, 3: 19, 4: 18}


def wis_pointwise(pred_lower, pred_upper, true, alpha):
    pred_lower = np.asarray(pred_lower, dtype=float)
    pred_upper = np.asarray(pred_upper, dtype=float)
    true = np.asarray(true, dtype=float)

    width = pred_upper - pred_lower
    below = (true < pred_lower) * (pred_lower - true)
    above = (true > pred_upper) * (true - pred_upper)
    return width + (2.0 / alpha) * below + (2.0 / alpha) * above


def normalize_date_key(date_series):
    """Normalize mixed date formats to YYYY-MM-DD keys for merging."""
    s = date_series.astype(str).str.strip()
    parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")

    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            s.loc[missing], format="%d/%m/%Y", errors="coerce"
        )

    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            s.loc[missing], dayfirst=True, errors="coerce"
        )

    return parsed.dt.strftime("%Y-%m-%d")


def find_true_col(df):
    return next((c for c in df.columns if str(c).strip().lower() == "true"), None)


def find_interval_cols(df):
    lower_col = next(
        (
            c
            for c in df.columns
            if "lower" in c.lower() or "0.1" in c or c.lower().startswith("lo")
        ),
        None,
    )
    upper_col = next(
        (
            c
            for c in df.columns
            if "upper" in c.lower() or "0.9" in c or c.lower().startswith("hi")
        ),
        None,
    )
    return lower_col, upper_col


def compute_and_save(country, step, model_name, alpha=0.2):
    target_dir = f"results/{model_name}_real_{country}_ILI"
    target_path = f"{target_dir}/rolling_pred_step{step}.csv"
    naive_path = f"results/Naive_real_{country}_ILI/rolling_pred_step{step}.csv"

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Missing prediction file: {target_path}")
    if not os.path.exists(naive_path):
        raise FileNotFoundError(f"Missing Naive truth file: {naive_path}")

    df_target = pd.read_csv(target_path)
    df_naive = pd.read_csv(naive_path)

    if "date" not in df_target.columns or "date" not in df_naive.columns:
        raise ValueError("Missing date column in prediction or Naive file.")

    df_target["date_key"] = normalize_date_key(df_target["date"])
    df_naive["date_key"] = normalize_date_key(df_naive["date"])

    bad_target_dates = df_target.loc[df_target["date_key"].isna(), "date"].head(5).tolist()
    bad_naive_dates = df_naive.loc[df_naive["date_key"].isna(), "date"].head(5).tolist()
    if bad_target_dates or bad_naive_dates:
        raise ValueError(
            f"Failed to parse dates: target={bad_target_dates}, naive={bad_naive_dates}"
        )

    true_col = find_true_col(df_naive)
    if not true_col:
        raise ValueError("Could not find TRUE/true column in Naive file.")

    lower_col, upper_col = find_interval_cols(df_target)
    if not lower_col or not upper_col:
        raise ValueError(
            f"Could not find lower/upper columns. Columns: {list(df_target.columns)}"
        )

    df_naive_truth = df_naive[["date_key", true_col]].drop_duplicates("date_key", keep="last")
    df_merged = pd.merge(
        df_target[["date_key", lower_col, upper_col]],
        df_naive_truth,
        on="date_key",
        how="left",
    )

    target_len = TARGET_LENGTHS.get(step, 21)
    if len(df_merged) > target_len:
        df_merged = df_merged.tail(target_len).copy()

    lower = df_merged[lower_col].to_numpy(dtype=float)
    upper = df_merged[upper_col].to_numpy(dtype=float)
    true = df_merged[true_col].to_numpy(dtype=float)

    mask = np.isfinite(true) & np.isfinite(lower) & np.isfinite(upper)
    true_clean = true[mask]
    lower_clean = lower[mask]
    upper_clean = upper[mask]

    if true_clean.size == 0:
        raise ValueError("No valid aligned true/lower/upper rows found.")

    if np.any(upper_clean < lower_clean):
        idx = np.where(upper_clean < lower_clean)[0][:5]
        raise ValueError(f"upper is smaller than lower at rows: {idx}")

    wis_points = wis_pointwise(lower_clean, upper_clean, true_clean, alpha=alpha)

    os.makedirs(target_dir, exist_ok=True)
    out_path = f"{target_dir}/wis80_point_step{step}.npy"
    np.save(out_path, wis_points)

    mean_wis = float(np.mean(wis_points))
    picp = float(np.mean((true_clean >= lower_clean) & (true_clean <= upper_clean)))
    mean_width = float(np.mean(upper_clean - lower_clean))

    return out_path, mean_wis, picp, mean_width, len(wis_points)


def main():
    countries = [
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
    alpha = 0.2

    models = ["Ensemble"]  # "Ensemble", "Respicast", "TabPFN_ts", "SEIR"

    for model in models:
        print("\n==========================================")
        print(f"Start computing model: {model}")
        print("==========================================")

        for country in countries:
            print(f"\n=== {country} ===")
            for step in [1, 2, 3, 4]:
                try:
                    out_path, mean_wis, picp, width, n = compute_and_save(
                        country, step, model, alpha=alpha
                    )
                    print(
                        f"step{step}: saved {out_path} | meanWIS80={mean_wis:.4f} "
                        f"| PICP80={picp:.3f} | width={width:.3f} | n={n}"
                    )
                except Exception as e:
                    print(f"step{step}: [skip] {e}")


if __name__ == "__main__":
    main()
