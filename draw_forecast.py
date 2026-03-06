import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.tools import visual

COUNTRY_YLIM = {
    "Belgium": (0, 2000),
    "Czechia": (0, 400),
    "Denmark": (0, 500),
    "France": (0, 1000),
    "Ireland": (0, 120),
    "Italy": (0, 2500),
    "Netherlands": (0, 200),
    "Poland": (0, 1400),
    "Romania": (0, 100),
}
# -----------------------------
# Helpers
# -----------------------------
def pick_date_col(df: pd.DataFrame):
    for c in ["date", "Date", "timestamp", "ds", "time"]:
        if c in df.columns:
            return c
    return None


def axis_flags_for_paper(model: str, split: str):
    m = str(model).strip().lower()
    s = str(split).strip().lower()
    # ✅ simulated 专用：Naive/DLinear/Autoformer 显示 y 轴；Autoformer/TabPFN 显示 x 轴
    if s == "simulated":
        show_yticks = m in {"naive", "dlinear", "autoformer"}
        show_xticks = m in {"autoformer", "tabpfn_ts"}
        show_ylabel = show_yticks
        show_xlabel = show_xticks
        return show_yticks, show_xticks, show_ylabel, show_xlabel
    # top strip real-only: only ARIMA_real shows y-axis ticks+label
    if m in {"arima", "tabpfn_ts", "Respicast", "naive"}:
        show_yticks = (m == "arima" and s == "real")
        show_xticks = False
        show_ylabel = show_yticks     # ✅ y label 跟着 y ticks
        show_xlabel = False
        return show_yticks, show_xticks, show_ylabel, show_xlabel

    # 3x3 block: y only left col (DLinear), x only bottom row (combined)
    show_yticks = (m == "dlinear")
    show_xticks = (s == "combined")
    show_ylabel = show_yticks        # ✅ 左列三张都有 Incidence
    show_xlabel = show_xticks        # ✅ 底行三张都有 Date
    return show_yticks, show_xticks, show_ylabel, show_xlabel


def ensure_datetime_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a datetime Series from df.
    Supports:
      - a single date-like column (date/timestamp/...)
      - anno + settimana (ISO week -> Monday)
    """
    date_col = pick_date_col(df)
    if date_col is not None:
        return pd.to_datetime(df[date_col])

    if ("anno" in df.columns) and ("settimana" in df.columns):
        yy = df["anno"].astype(int).to_numpy()
        ww = df["settimana"].astype(int).to_numpy()
        dates = [pd.Timestamp.fromisocalendar(int(y), int(w), 1) for y, w in zip(yy, ww)]
        return pd.to_datetime(pd.Series(dates))

    raise ValueError(
        f"No date column found. Expected date/timestamp/etc OR anno+settimana. "
        f"Found columns: {list(df.columns)}"
    )


def find_step_csv(folder: Path, k: int):
    # accept: step1.csv / step_1.csv / rolling_pred_step1.csv / tabpfn_ts_pred_step1.csv ...
    pats = [f"*step{k}*.csv", f"*step_{k}*.csv"]
    hits = []
    for pat in pats:
        hits += list(folder.glob(pat))
    hits = sorted(set(hits))
    return hits[0] if hits else None


def parse_model_data_country(folder_name: str):
    """
    folder_name examples:
      TabPFN_ts_real_Belgium_ILI
      TabPFN_ts_simulated_Italy_ILI_median
      DLinear_real_Ireland_ILI
      Respicast_real_Poland_ILI
      ARIMA_simulated_Poland_ILI_median
      Autoformer_augmented_France_ILI
    Return (model, split, data_str, country)
    """
    if folder_name.startswith("TabPFN_ts_"):
        model = "TabPFN_ts"
        data_str = folder_name[len("TabPFN_ts_") :]
    else:
        if "_" not in folder_name:
            return folder_name, None, None, None
        model, data_str = folder_name.split("_", 1)

    m = re.match(r"^(real|simulated|augmented|combined)_(.+?)_ILI", data_str)
    split = m.group(1) if m else None
    country = m.group(2) if m else None
    return model, split, data_str, country


def to_series(vals, idx: pd.DatetimeIndex):
    if vals is None:
        return None
    arr = np.asarray(vals, dtype=float)
    return pd.Series(arr, index=idx)


def call_visual(true_s, pred_s, save_path: Path, *, lower_s=None, upper_s=None, seq_len=4,
                paper=True,
                show_xticks=True, show_yticks=True,
                show_xlabel=True, show_ylabel=True,
                ylim=None):
    """
    show_xticks/show_yticks: 控制刻度文本（日期/数字）
    show_xlabel/show_ylabel: 控制轴名称（Date/Incidence）
    """
    kwargs = dict(lower=lower_s, upper=upper_s, seq_len=seq_len)

    try:
        return visual(
            true_s, pred_s, str(save_path),
            **kwargs,
            paper=paper,
            show_xticklabels=show_xticks,
            show_yticklabels=show_yticks,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            ylim=ylim
        )
    except TypeError:
        return visual(true_s, pred_s, str(save_path), **kwargs)


# -----------------------------
# TabPFN pipeline
# -----------------------------
def load_dataset_real_last25(dataset_dir: Path, country: str, last_n: int = 25):
    path = dataset_dir / f"real_{country}_ILI.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path)
    dates = ensure_datetime_series(df)
    if "incidenza" not in df.columns:
        raise ValueError(f"'incidenza' not found in {path.name}. cols={list(df.columns)}")

    df["_date"] = dates
    df = df.tail(last_n).copy()

    idx = pd.DatetimeIndex(pd.to_datetime(df["_date"]))
    y_true = df["incidenza"].astype(float).to_numpy()
    return idx, y_true


def load_tabpfn_pred_last25(tabpfn_folder: Path, k: int, last_n: int = 25):
    p = tabpfn_folder / f"tabpfn_ts_pred_step{k}.csv"
    if not p.exists():
        p2 = tabpfn_folder / f"tabpfn_ts_pred_step_{k}.csv"
        if p2.exists():
            p = p2
        else:
            raise FileNotFoundError(f"Missing TabPFN pred file: {p}")

    df = pd.read_csv(p)
    dates = ensure_datetime_series(df)
    df["_date"] = dates
    df = df.tail(last_n).copy()

    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    med = pick(["0.5", "q0.5", "p50", "median"])
    lo = pick(["0.1", "q0.1", "p10"])
    hi = pick(["0.9", "q0.9", "p90"])
    if med is None:
        raise ValueError(f"No median col in {p.name}. cols={list(df.columns)}")

    y_pred = df[med].astype(float).to_numpy()
    lower = df[lo].astype(float).to_numpy() if lo else None
    upper = df[hi].astype(float).to_numpy() if hi else None
    return y_pred, lower, upper


def draw_tabpfn(results_folder: Path, dataset_dir: Path, out_root: Path, country: str,
               split: str = "real", last_n: int = 25, seq_len: int = 4):
    out_root.mkdir(parents=True, exist_ok=True)

    idx, y_true = load_dataset_real_last25(dataset_dir, country, last_n=last_n)
    true_s = to_series(y_true, idx)

    # TabPFN pred only on future part
    idx_pred = idx[int(seq_len):]  # 25-4=21


    for k in [1, 2, 3, 4]:
        y_pred, lower, upper = load_tabpfn_pred_last25(results_folder, k, last_n=last_n)

        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_pred) == len(idx):
            y_pred = y_pred[int(seq_len):]
        elif len(y_pred) != len(idx_pred):
            raise ValueError(f"TabPFN step{k}: pred length {len(y_pred)} != expected {len(idx_pred)}")

        pred_s = pd.Series(y_pred, index=idx_pred)

        lower_s = None
        upper_s = None
        if lower is not None and upper is not None:
            lo = np.asarray(lower, dtype=float)
            hi = np.asarray(upper, dtype=float)
            if len(lo) == len(idx):
                lo = lo[int(seq_len):]
                hi = hi[int(seq_len):]
            if len(lo) == len(idx_pred) and len(hi) == len(idx_pred):
                lower_s = pd.Series(lo, index=idx_pred)
                upper_s = pd.Series(hi, index=idx_pred)

        save_path = out_root / f"rolling_test_step{k}.png"
        show_yticks, show_xticks, show_ylabel, show_xlabel = axis_flags_for_paper("TabPFN_ts", split)
        ylim = COUNTRY_YLIM.get(country)
        call_visual(
            true_s, pred_s, save_path,
            lower_s=lower_s, upper_s=upper_s, seq_len=seq_len,
            paper=False,
            show_xticks=show_xticks, show_yticks=show_yticks,
            show_xlabel=show_xlabel, show_ylabel=show_ylabel,
            ylim=ylim
        )
        print(f"[OK] {save_path}")


# -----------------------------
# Respicast pipeline (real-only top strip)
# -----------------------------
def load_respicast_pred_last25(respicast_folder: Path, k: int, last_n: int = 25):
    p = respicast_folder / f"rolling_pred_step{k}.csv"
    if not p.exists():
        p2 = respicast_folder / f"rolling_pred_step_{k}.csv"
        if p2.exists():
            p = p2
        else:
            raise FileNotFoundError(f"Missing Respicast pred file: {p}")

    df = pd.read_csv(p)
    dates = ensure_datetime_series(df)
    df["_date"] = dates
    df = df.sort_values("_date").tail(last_n).copy()

    if "target" not in df.columns:
        raise ValueError(f"No 'target' col in {p.name}. cols={list(df.columns)}")

    y_pred = df["target"].astype(float).to_numpy()
    lower = df["lower80"].astype(float).to_numpy() if "lower80" in df.columns else None
    upper = df["upper80"].astype(float).to_numpy() if "upper80" in df.columns else None

    idx_pred = pd.DatetimeIndex(pd.to_datetime(df["_date"]))
    return idx_pred, y_pred, lower, upper


def draw_respicast(results_folder: Path, dataset_dir: Path, out_root: Path, country: str,
                  split: str = "real", last_n: int = 25, seq_len: int = 4):
    out_root.mkdir(parents=True, exist_ok=True)

    idx_true, y_true = load_dataset_real_last25(dataset_dir, country, last_n=last_n)
    true_s = to_series(y_true, idx_true)


    for k in [1, 2, 3, 4]:
        idx_pred, y_pred, lower, upper = load_respicast_pred_last25(results_folder, k, last_n=last_n)

        pred_s = pd.Series(np.asarray(y_pred, dtype=float), index=idx_pred)
        lower_s = pd.Series(lower, index=idx_pred) if lower is not None else None
        upper_s = pd.Series(upper, index=idx_pred) if upper is not None else None

        save_path = out_root / f"rolling_test_step{k}.png"
        show_yticks, show_xticks, show_ylabel, show_xlabel = axis_flags_for_paper("Respicast", split)
        ylim = COUNTRY_YLIM.get(country)
        call_visual(
            true_s, pred_s, save_path,
            lower_s=lower_s, upper_s=upper_s, seq_len=seq_len,
            paper=False,
            show_xticks=show_xticks, show_yticks=show_yticks,
            show_xlabel=show_xlabel, show_ylabel=show_ylabel,
            ylim=ylim
        )
        print(f"[OK] {save_path}")


# -----------------------------
# Generic pipeline
# -----------------------------
def pick_true_col(df: pd.DataFrame):
    for c in ["true", "y_true", "label", "gt", "incidenza"]:
        if c in df.columns:
            return c
    return None


def pick_pred_col(df: pd.DataFrame, k: int):
    cands = [f"pred_step{k}", f"y_pred_step{k}", f"pred{k}", f"step{k}",
             "pred", "y_pred", "mean", "mu"]
    for c in cands:
        if c in df.columns:
            return c
    for c in ["0.5", "q0.5", "p50", "median"]:
        if c in df.columns:
            return c
    return None


def pick_interval_cols(df: pd.DataFrame, k: int):
    lows = [f"lower80_step{k}", f"lo80_step{k}", f"lower_step{k}", f"lower{k}", "lower", "lo"]
    ups  = [f"upper80_step{k}", f"hi80_step{k}", f"upper_step{k}", f"upper{k}", "upper", "hi"]
    low = next((c for c in lows if c in df.columns), None)
    up  = next((c for c in ups if c in df.columns), None)
    if low and up:
        return low, up

    qlow = next((c for c in ["0.1", "q0.1", "p10"] if c in df.columns), None)
    qup  = next((c for c in ["0.9", "q0.9", "p90"] if c in df.columns), None)
    if qlow and qup:
        return qlow, qup
    return None, None


def draw_generic(results_folder: Path, out_root: Path, model: str, split: str, country: str = None,
                 last_n: int = 25, seq_len: int = 4):
    out_root.mkdir(parents=True, exist_ok=True)


    for k in [1, 2, 3, 4]:
        csv_path = find_step_csv(results_folder, k)
        if csv_path is None:
            print(f"[WARN] Missing step{k} csv in {results_folder.name} (skip)")
            continue

        df = pd.read_csv(csv_path)
        dates = ensure_datetime_series(df)
        df["_date"] = dates
        df = df.tail(last_n).copy()
        idx = pd.DatetimeIndex(pd.to_datetime(df["_date"]))

        true_col = pick_true_col(df)
        pred_col = pick_pred_col(df, k)
        low_col, up_col = pick_interval_cols(df, k)

        if true_col is None or pred_col is None:
            print(f"[WARN] Missing true/pred in {csv_path.name} (skip). cols={list(df.columns)}")
            continue

        true_s = pd.Series(df[true_col].astype(float).to_numpy(), index=idx)
        pred_s = pd.Series(df[pred_col].astype(float).to_numpy(), index=idx)
        lower_s = pd.Series(df[low_col].astype(float).to_numpy(), index=idx) if low_col else None
        upper_s = pd.Series(df[up_col].astype(float).to_numpy(), index=idx) if up_col else None

        save_path = out_root / f"rolling_test_step{k}.png"
        show_yticks, show_xticks, show_ylabel, show_xlabel = axis_flags_for_paper(model, split)
        ylim = COUNTRY_YLIM.get(country)  # country 是 parse_model_data_country() 解析出来的
        call_visual(
            true_s, pred_s, save_path,
            lower_s=lower_s, upper_s=upper_s, seq_len=seq_len,
            paper=False,
            show_xticks=show_xticks, show_yticks=show_yticks,
            show_xlabel=show_xlabel, show_ylabel=show_ylabel,
            ylim=ylim
        )
        print(f"[OK] {save_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    repo_root = Path(".").resolve()
    if (repo_root / "epi4cast" / "results").exists() and not (repo_root / "results").exists():
        repo_root = repo_root / "epi4cast"

    results_dir = repo_root / "results"
    dataset_dir = repo_root / "dataset"
    out_base = repo_root / "test_results"

    if not results_dir.exists():
        raise FileNotFoundError(f"Missing: {results_dir}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing: {dataset_dir}")

    folders = sorted([p for p in results_dir.iterdir() if p.is_dir()])
    folders = [p for p in folders if "_simulated_" in p.name]
    if not folders:
        raise FileNotFoundError(f"No folders under: {results_dir}")

    for f in folders:
        model, split, data_str, country = parse_model_data_country(f.name)
        out_root = out_base / f.name

        if model == "TabPFN_ts":
            if country is None:
                print(f"[WARN] Cannot parse country from {f.name}, skip TabPFN.")
                continue
            draw_tabpfn(f, dataset_dir, out_root, country=country, split=split or "real", last_n=25, seq_len=4)

        elif str(model).strip().lower() == "Respicast":
            if country is None:
                print(f"[WARN] Cannot parse country from {f.name}, skip Respicast.")
                continue
            draw_respicast(f, dataset_dir, out_root, country=country, split=split or "real", last_n=25, seq_len=4)

        else:
            # generic: includes ARIMA_real (top strip) and DLinear/LSTM/Autoformer (3x3 block)
            draw_generic(f, out_root, model=model, split=split or "real", country=country, last_n=25, seq_len=4)


if __name__ == "__main__":
    main()