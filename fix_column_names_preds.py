# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import re
from pathlib import Path
import pandas as pd

ROOT = Path("./results")

# ==========================================
# 1. Settings
# ==========================================
NEW_DATES = [
    "11/11/2024", "18/11/2024", "25/11/2024", "02/12/2024", "09/12/2024",
    "16/12/2024", "23/12/2024", "30/12/2024", "06/01/2025", "13/01/2025",
    "20/01/2025", "27/01/2025", "03/02/2025", "10/02/2025", "17/02/2025",
    "24/02/2025", "03/03/2025", "10/03/2025", "17/03/2025", "24/03/2025",
    "31/03/2025"
]
TARGET_LEN = 21

PRED_NAMES = {"pred", "target"}
PRED_STEP_RE = re.compile(r"^pred_step\d+$", re.IGNORECASE)
DATE_NAMES = {"timestamp", "time", "datetime", "date_time"}
TABPFN_FILE_RE = re.compile(r"^tabpfn_ts_pred_step(\d+)\.csv$", re.IGNORECASE)

def coalesce_cols(df: pd.DataFrame, cols):
    out = None
    for c in cols:
        s = df[c]
        if out is None: out = s.copy()
        else: out = out.where(out.notna(), s)
    return out

def is_seir_path(path: Path):
    return any(str(part).lower().startswith("seir_") for part in path.parts)

# ==========================================
# Step 1：Change TabPFN filename
# ==========================================
def step1_rename_files(root_dir: Path):
    print("🚀 --- Step 1: Rename TabPFN files ---")
    for p in root_dir.rglob("*.csv"):
        match = TABPFN_FILE_RE.match(p.name)
        if match:
            new_path = p.with_name(f"rolling_pred_step{match.group(1)}.csv")
            if not new_path.exists():
                p.rename(new_path)
                print(f"   [rename] {p.name} -> {new_path.name}")
    print("✅ Step 1 completed.\n")

# ==========================================
# Step 2 & 3：change column names + substitute 21 dates
# ==========================================
def step2_and_3_process_csv(path: Path):
    if "ensemble" in str(path).lower():
        return "  ensemble pass"

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return f"❌ can't read: {e}"

    modified = False
    log_msgs = []

    # 1. name timestamp 'date'
    rename_dict = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in DATE_NAMES or (cl == "date" and c != "date"):
            rename_dict[c] = "date"
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
        modified = True

    # 2. name prediction columns 'pred'
    cand = [c for c in df.columns if str(c).strip().lower() in PRED_NAMES or PRED_STEP_RE.match(str(c).strip().lower())]
    if is_seir_path(path):
        median_cols = [c for c in df.columns if str(c).strip().lower() == "median"]
        if median_cols:
            pred_cols = [c for c in cand if str(c).strip().lower() == "pred"]
            cand = median_cols + pred_cols

    if cand and not (len(cand) == 1 and str(cand[0]).strip() == "pred"):
        df["pred"] = coalesce_cols(df, cand)
        for c in cand:
            if str(c).strip() != "pred": df.drop(columns=[c], inplace=True)
        modified = True

    if modified: log_msgs.append("column name fixed")

    # 3. Core:substitute 21 rows of dates
    if "date" in df.columns:
        if len(df) >= TARGET_LEN:
            col_idx = df.columns.get_loc("date")
            df.iloc[-TARGET_LEN:, col_idx] = NEW_DATES
            modified = True
            log_msgs.append("dates replaced (last 21 rows)")
        else:
            log_msgs.append(f"⚠️ row count ({len(df)}) insufficient, cannot replace dates")

    # 4. save
    if modified:
        cols = list(df.columns)
        if "date" in cols and cols.index("date") != 0:
            cols.remove("date"); cols.insert(0, "date")
        if "pred" in cols and "TRUE" in cols:
            cols.remove("pred"); true_idx = cols.index("TRUE")
            cols.insert(true_idx + 1, "pred")
            
        df = df[cols]
        df.to_csv(path, index=False)
        return " + ".join(log_msgs)
        
    return "Done"

def main():
    if not ROOT.exists(): return print("❌ Error: Folder not found")
    step1_rename_files(ROOT)
    print(" --- Step 2 & 3: Process column names and dates ---")
    for p in ROOT.rglob("*.csv"):
        res = step2_and_3_process_csv(p)
        if "❌" in res or "⚠️" in res: print(f"   [Warning/Error] {p.name} -> {res}")
        elif "无需" not in res and "跳过" not in res: print(f"   [Updated] {p.name} -> {res}")
    print("\n🎉 Step 2 & 3 completed!")

if __name__ == "__main__":
    main()
