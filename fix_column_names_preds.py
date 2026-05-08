import re
from pathlib import Path
import pandas as pd

ROOT = Path("./results")

# ==========================================
# 1. 全局配置
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
# 步骤 1：先改 TabPFN 文件名
# ==========================================
def step1_rename_files(root_dir: Path):
    print("🚀 --- 步骤 1: 开始重命名 TabPFN 文件 ---")
    for p in root_dir.rglob("*.csv"):
        match = TABPFN_FILE_RE.match(p.name)
        if match:
            new_path = p.with_name(f"rolling_pred_step{match.group(1)}.csv")
            if not new_path.exists():
                p.rename(new_path)
                print(f"   [改名] {p.name} -> {new_path.name}")
    print("✅ 步骤 1 完成。\n")

# ==========================================
# 步骤 2 & 3：改列名 + 暴力替换最后 21 行日期
# ==========================================
def step2_and_3_process_csv(path: Path):
    # 如果是 ensemble 直接跳过
    if "ensemble" in str(path).lower():
        return "⏭️ 属于 ensemble，已跳过"

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return f"❌ 无法读取: {e}"

    modified = False
    log_msgs = []

    # 1. 统一时间列名为 'date'
    rename_dict = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in DATE_NAMES or (cl == "date" and c != "date"):
            rename_dict[c] = "date"
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
        modified = True

    # 2. 统一预测列名为 'pred'
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

    if modified: log_msgs.append("列名已统一")

    # 3. 核心：暴力替换最后 21 行的日期
    if "date" in df.columns:
        if len(df) >= TARGET_LEN:
            col_idx = df.columns.get_loc("date")
            df.iloc[-TARGET_LEN:, col_idx] = NEW_DATES
            modified = True
            log_msgs.append("日期已覆盖(最后21行)")
        else:
            log_msgs.append(f"⚠️ 行数({len(df)})不足21，无法替换日期")

    # 4. 保存
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
        
    return "✅ 无需修改"

def main():
    if not ROOT.exists(): return print("❌ 错误：找不到文件夹")
    step1_rename_files(ROOT)
    print("🚀 --- 步骤 2 & 3: 处理列名与日期 ---")
    for p in ROOT.rglob("*.csv"):
        res = step2_and_3_process_csv(p)
        if "❌" in res or "⚠️" in res: print(f"   [警告/错误] {p.name} -> {res}")
        elif "无需" not in res and "跳过" not in res: print(f"   [更新] {p.name} -> {res}")
    print("\n🎉 全部处理完毕！")

if __name__ == "__main__":
    main()
