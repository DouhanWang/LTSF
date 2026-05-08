# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import os
import re
import pandas as pd
import numpy as np

# ========== 配置 ==========
INPUT_CSV = "./dataset/ILI_4seasons_w42_w14_combined.csv"   
OUT_DIR = "per_country_csv"                        # output folder

SEASON_ORDER = ["2017-2018", "2018-2019", "2023-2024", "2024-2025"]
SEASON_ID = {s: i for i, s in enumerate(SEASON_ORDER)}  # season_id: 0,1,2,3
# =========================

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# Analyze year_week -> anno(year), settimana(week)
m = df["year_week"].astype(str).str.extract(r"(?P<anno>\d{4})-W(?P<settimana>\d{1,2})")
df["anno"] = m["anno"].astype(int)
df["settimana"] = m["settimana"].astype(int)


if "season" not in df.columns:
    df["season"] = np.where(
        df["settimana"] >= 42,
        df["anno"].astype(str) + "-" + (df["anno"] + 1).astype(str),
        (df["anno"] - 1).astype(str) + "-" + df["anno"].astype(str),
    )

# 4 season、week42-52 + week1-14
df = df[df["season"].isin(SEASON_ORDER)].copy()
df = df[((df["settimana"].between(42, 52)) | (df["settimana"].between(1, 14)))].copy()


df["incidenza"] = df["value"]
df["season_id"] = df["season"].map(SEASON_ID)


df = df.sort_values(["location", "season_id", "anno", "settimana"]).reset_index(drop=True)


for idx, (loc, g) in enumerate(df.groupby("location", sort=True), start=0):
    out = g[["season_id", "anno", "settimana", "incidenza"]].copy()
    out.insert(0, "item_id", 0)  

    out_path = os.path.join(OUT_DIR, f"{loc}.csv")
    out.to_csv(out_path, index=False)

print(f"Done. Saved {df['location'].nunique()} files to: {OUT_DIR}")
