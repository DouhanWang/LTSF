import os
import re
import pandas as pd
import numpy as np

# ========== 配置 ==========
INPUT_CSV = "./dataset/ILI_4seasons_w42_w14_combined.csv"   # 改成你的输入文件
OUT_DIR = "per_country_csv"                        # 输出文件夹
# 你要的四个 season（按顺序）
SEASON_ORDER = ["2017-2018", "2018-2019", "2023-2024", "2024-2025"]
SEASON_ID = {s: i for i, s in enumerate(SEASON_ORDER)}  # season_id: 0,1,2,3
# =========================

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# 解析 year_week -> anno(年份), settimana(周)
m = df["year_week"].astype(str).str.extract(r"(?P<anno>\d{4})-W(?P<settimana>\d{1,2})")
df["anno"] = m["anno"].astype(int)
df["settimana"] = m["settimana"].astype(int)

# 如果没有 season 列，就按 week>=42 判断 season（和我们之前逻辑一致）
if "season" not in df.columns:
    df["season"] = np.where(
        df["settimana"] >= 42,
        df["anno"].astype(str) + "-" + (df["anno"] + 1).astype(str),
        (df["anno"] - 1).astype(str) + "-" + df["anno"].astype(str),
    )

# 只保留你要的四个 season、week42-52 + week1-14
df = df[df["season"].isin(SEASON_ORDER)].copy()
df = df[((df["settimana"].between(42, 52)) | (df["settimana"].between(1, 14)))].copy()

# 生成你要的列名
df["incidenza"] = df["value"]
df["season_id"] = df["season"].map(SEASON_ID)

# 排序：先按 season_id，再按 anno，再按 settimana（保证 W42-52 在前，W1-14 在后）
df = df.sort_values(["location", "season_id", "anno", "settimana"]).reset_index(drop=True)

# 按国家导出
for idx, (loc, g) in enumerate(df.groupby("location", sort=True), start=0):
    out = g[["season_id", "anno", "settimana", "incidenza"]].copy()
    out.insert(0, "item_id", 0)  # 截图里 item_id 全是 0；若你想每国不同可改成 idx

    out_path = os.path.join(OUT_DIR, f"{loc}.csv")
    out.to_csv(out_path, index=False)

print(f"Done. Saved {df['location'].nunique()} files to: {OUT_DIR}")
