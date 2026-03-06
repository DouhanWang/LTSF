import re
from pathlib import Path
import pandas as pd

ROOT = Path("results")

# 需要统一成 pred 的候选列名
PRED_NAMES = {"pred", "target"}
PRED_STEP_RE = re.compile(r"^pred_step\d+$", re.IGNORECASE)

def coalesce_cols(df: pd.DataFrame, cols):
    """按顺序合并多个列：优先保留非空值"""
    out = None
    for c in cols:
        s = df[c]
        if out is None:
            out = s.copy()
        else:
            out = out.where(out.notna(), s)
    return out

def fix_one_csv(path: Path) -> bool:
    df = pd.read_csv(path)

    # 找候选预测列
    cand = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in PRED_NAMES or PRED_STEP_RE.match(cl):
            cand.append(c)

    # 没有候选列就不改
    if not cand:
        return False

    # 如果已经只有一个 pred 且没有其它候选，不改
    if len(cand) == 1 and str(cand[0]).strip().lower() == "pred":
        return False

    # 生成统一的 pred（合并所有候选列）
    df["pred"] = coalesce_cols(df, cand)

    # 删除旧候选列（除了 pred 本身）
    for c in cand:
        if str(c).strip().lower() != "pred":
            df.drop(columns=[c], inplace=True)

    # 把 pred 放到 TRUE 后面（如果 TRUE 存在）
    cols = list(df.columns)
    if "TRUE" in df.columns:
        cols.remove("pred")
        true_idx = cols.index("TRUE")
        cols.insert(true_idx + 1, "pred")
        df = df[cols]

    df.to_csv(path, index=False)
    return True

def main():
    changed = 0
    total = 0
    for p in ROOT.rglob("*.csv"):
        total += 1
        try:
            if fix_one_csv(p):
                changed += 1
                print(f"[OK] fixed: {p}")
        except Exception as e:
            print(f"[WARN] failed: {p} | {e}")

    print(f"Done. CSV scanned={total}, modified={changed}")

if __name__ == "__main__":
    main()