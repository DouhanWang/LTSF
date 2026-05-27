# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results")
CSV_PATTERNS = ("rolling_pred_step*.csv", "tabpfn_ts_pred_step*.csv")
NPY_PATTERNS = ("pred_step*.npy", "lower80_step*.npy", "upper80_step*.npy")

PRED_COLS = ("pred", "median", "mean", "0.5")
LOWER_COLS = ("lower80", "lower", "lo", "0.1")
UPPER_COLS = ("upper80", "upper", "hi", "0.9")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip negative forecast outputs to zero for all model result folders."
    )
    parser.add_argument("--root", default=str(ROOT), help="Results root folder.")
    parser.add_argument("--dry-run", action="store_true", help="Report files that would change without writing them.")
    parser.add_argument(
        "--no-npy",
        action="store_true",
        help="Only process prediction CSV files; leave pred/lower/upper .npy files unchanged.",
    )
    return parser.parse_args()


def extract_step(path: Path) -> int | None:
    match = re.search(r"step_?(\d+)", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    by_lower = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        col = by_lower.get(candidate.lower())
        if col is not None:
            return col
    return None


def prediction_col(df: pd.DataFrame, step: int | None) -> str | None:
    candidates = []
    if step is not None:
        candidates.append(f"pred_step{step}")
    candidates.extend(PRED_COLS)

    col = first_existing_col(df, candidates)
    if col is not None:
        return col

    # Some external forecast exports use target as the prediction column.
    # Do not use it when a true/ground-truth column exists.
    true_like = {"true", "truth", "ground_truth", "groundtruth", "actual", "incidenza"}
    if not any(str(c).lower() in true_like for c in df.columns):
        return first_existing_col(df, ["target"])

    return None


def interval_cols(df: pd.DataFrame, step: int | None) -> tuple[str | None, str | None]:
    lower_candidates = []
    upper_candidates = []
    if step is not None:
        lower_candidates.append(f"lower80_step{step}")
        upper_candidates.append(f"upper80_step{step}")

    lower_candidates.extend(LOWER_COLS)
    upper_candidates.extend(UPPER_COLS)

    return first_existing_col(df, lower_candidates), first_existing_col(df, upper_candidates)


def quantile_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        text = str(col).strip()
        try:
            q = float(text)
        except ValueError:
            continue
        if 0.0 < q < 1.0:
            cols.append(col)
    return cols


def clip_numeric_series(series: pd.Series) -> tuple[pd.Series, int]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float).copy()
    mask = np.isfinite(values) & (values < 0.0)
    changed = int(mask.sum())
    values[mask] = 0.0
    return pd.Series(values, index=series.index), changed


def clip_csv(df: pd.DataFrame, step: int | None) -> tuple[bool, int]:
    pred_col = prediction_col(df, step)
    lower_col, upper_col = interval_cols(df, step)
    q_cols = quantile_cols(df)

    cols_to_clip = []
    for col in [pred_col, lower_col, upper_col, *q_cols]:
        if col is not None and col not in cols_to_clip:
            cols_to_clip.append(col)

    if not cols_to_clip:
        return False, 0

    total_changed = 0
    for col in cols_to_clip:
        df[col], changed = clip_numeric_series(df[col])
        total_changed += changed

    if pred_col is not None and lower_col is not None:
        pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float).copy()
        low = pd.to_numeric(df[lower_col], errors="coerce").to_numpy(dtype=float).copy()
        mask = np.isfinite(pred) & np.isfinite(low) & (low > pred)
        total_changed += int(mask.sum())
        low[mask] = pred[mask]
        df[lower_col] = low

    if pred_col is not None and upper_col is not None:
        pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float).copy()
        upper = pd.to_numeric(df[upper_col], errors="coerce").to_numpy(dtype=float).copy()
        mask = np.isfinite(pred) & np.isfinite(upper) & (upper < pred)
        total_changed += int(mask.sum())
        upper[mask] = pred[mask]
        df[upper_col] = upper

    if lower_col is not None and upper_col is not None:
        low = pd.to_numeric(df[lower_col], errors="coerce").to_numpy(dtype=float).copy()
        upper = pd.to_numeric(df[upper_col], errors="coerce").to_numpy(dtype=float).copy()
        mask = np.isfinite(low) & np.isfinite(upper) & (upper < low)
        total_changed += int(mask.sum())
        upper[mask] = low[mask]
        df[upper_col] = upper

    return True, total_changed


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)


def process_csv(path: Path, dry_run: bool) -> tuple[bool, int]:
    step = extract_step(path)
    df = pd.read_csv(path)
    ok, changed = clip_csv(df, step)
    if not ok:
        return False, 0

    if changed > 0 and not dry_run:
        backup_once(path)
        df.to_csv(path, index=False)

    return True, changed


def clip_array(arr: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.asarray(arr, dtype=float).copy()
    mask = np.isfinite(out) & (out < 0.0)
    changed = int(mask.sum())
    out[mask] = 0.0
    return out, changed


def process_npy_group(folder: Path, step: int, dry_run: bool) -> int:
    paths = {
        "pred": folder / f"pred_step{step}.npy",
        "lower": folder / f"lower80_step{step}.npy",
        "upper": folder / f"upper80_step{step}.npy",
    }

    arrays: dict[str, np.ndarray] = {}
    total_changed = 0
    for key, path in paths.items():
        if not path.exists():
            continue
        clipped, changed = clip_array(np.load(path))
        arrays[key] = clipped
        total_changed += changed

    pred = arrays.get("pred")
    lower = arrays.get("lower")
    upper = arrays.get("upper")

    if pred is not None and lower is not None and pred.shape == lower.shape:
        mask = np.isfinite(pred) & np.isfinite(lower) & (lower > pred)
        total_changed += int(mask.sum())
        lower[mask] = pred[mask]

    if pred is not None and upper is not None and pred.shape == upper.shape:
        mask = np.isfinite(pred) & np.isfinite(upper) & (upper < pred)
        total_changed += int(mask.sum())
        upper[mask] = pred[mask]

    if lower is not None and upper is not None and lower.shape == upper.shape:
        mask = np.isfinite(lower) & np.isfinite(upper) & (upper < lower)
        total_changed += int(mask.sum())
        upper[mask] = lower[mask]

    if total_changed > 0 and not dry_run:
        for key, arr in arrays.items():
            backup_once(paths[key])
            np.save(paths[key], arr)

    return total_changed


def collect_csv_targets(root: Path) -> list[Path]:
    targets: list[Path] = []
    for pattern in CSV_PATTERNS:
        targets.extend(root.rglob(pattern))
    return sorted(set(targets))


def collect_npy_groups(root: Path) -> list[tuple[Path, int]]:
    groups = set()
    for pattern in NPY_PATTERNS:
        for path in root.rglob(pattern):
            step = extract_step(path)
            if step is not None:
                groups.add((path.parent, step))
    return sorted(groups)


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Cannot find folder: {root.resolve()}")

    csv_targets = collect_csv_targets(root)
    if not csv_targets:
        print(f"No prediction CSV files found under: {root.resolve()}")
    else:
        print(f"Found {len(csv_targets)} prediction CSV files under: {root.resolve()}")

    csv_processed = 0
    csv_changed_files = 0
    csv_changed_values = 0
    for path in csv_targets:
        ok, changed = process_csv(path, dry_run=args.dry_run)
        if not ok:
            print(f"[SKIP CSV] no forecast columns: {path}")
            continue

        csv_processed += 1
        csv_changed_values += changed
        if changed > 0:
            csv_changed_files += 1
            status = "WOULD FIX" if args.dry_run else "FIXED"
            print(f"[{status} CSV] changed={changed}: {path}")

    npy_changed_groups = 0
    npy_changed_values = 0
    if not args.no_npy:
        npy_groups = collect_npy_groups(root)
        print(f"Found {len(npy_groups)} npy step groups under: {root.resolve()}")
        for folder, step in npy_groups:
            changed = process_npy_group(folder, step, dry_run=args.dry_run)
            npy_changed_values += changed
            if changed > 0:
                npy_changed_groups += 1
                status = "WOULD FIX" if args.dry_run else "FIXED"
                print(f"[{status} NPY] changed={changed}: {folder} step{step}")

    mode = "dry run" if args.dry_run else "write"
    print(
        f"Done ({mode}). CSV processed={csv_processed}, changed_files={csv_changed_files}, "
        f"changed_values={csv_changed_values}; NPY changed_groups={npy_changed_groups}, "
        f"changed_values={npy_changed_values}."
    )


if __name__ == "__main__":
    main()
