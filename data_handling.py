import numpy as np
import pandas as pd
from datetime import date
npz_paths = [
    "dataset/simulations_ILI/FR_2017_2018.npz",
    "dataset/simulations_ILI/FR_2018_2019.npz",
    "dataset/simulations_ILI/FR_2023_2024.npz",
    "dataset/simulations_ILI/FR_2024_2025.npz",
]


def load_first_array(npz_path):
    z = np.load(npz_path)
    arr = z[z.files[0]]  # take the first key
    if arr.shape != (1000, 27):
        raise ValueError(f"{npz_path}: got {arr.shape}, expected (1000, 27)")
    return arr


def iso_weeks_in_year(y: int) -> int:
    # ISO weeks: the week containing Dec 28 is always the last ISO week of the year
    return date(y, 12, 28).isocalendar().week  # 52 or 53


def generate_iso_yearweeks(start_year: int, start_week: int, n_weeks: int):
    """Generate n_weeks (anno, settimana) starting from ISO year/week."""
    y, w = start_year, start_week
    out = []
    for _ in range(n_weeks):
        out.append((y, w))
        w += 1
        if w > iso_weeks_in_year(y):
            y += 1
            w = 1
    return out


season_starts = [
    (2017, 40),  # season_id=0
    (2018, 40),  # season_id=1
    (2023, 40),  # season_id=2
    (2024, 40),  # season_id=3
]

dfs = []
for season_id, (npz_path, (sy, sw)) in enumerate(zip(npz_paths, season_starts)):
    arr = load_first_array(npz_path)               # (1000, 27)
    wk = generate_iso_yearweeks(sy, sw, 27)         # 27个(anno,settimana)

    df = pd.DataFrame({
        "item_id": np.repeat(np.arange(1000), 27),
        "season_id": season_id,                    # ✅ 新增：标记season
        "anno": np.tile([a for a, w in wk], 1000),
        "settimana": np.tile([w for a, w in wk], 1000),
        "incidenza": arr.reshape(-1),
    })
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

combined_df = combined_df[~combined_df["settimana"].isin([40, 41])]

combined_df.to_csv("dataset/simulated_France_ILI.csv", index=False)