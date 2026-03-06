from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import re

ROOT = Path("results")  # 运行脚本时，确保当前工作目录是项目根目录(包含 results\ 的那个)

def clip_one_step(df: pd.DataFrame, step: int) -> bool:
    p = f"pred_step{step}"
    l = f"lower80_step{step}"
    u = f"upper80_step{step}"
    if not all(c in df.columns for c in (p, l, u)):
        return False

    # 防止被读成字符串
    for c in (p, l, u):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    pred = df[p].to_numpy(dtype=float)
    low  = df[l].to_numpy(dtype=float)
    up   = df[u].to_numpy(dtype=float)

    # 只改有限值，NaN 保持不动
    mp = np.isfinite(pred)
    ml = np.isfinite(low)
    mu = np.isfinite(up)

    # 1) 点预测非负
    pred[mp] = np.maximum(pred[mp], 0.0)

    # 2) 下界非负
    low[ml] = np.maximum(low[ml], 0.0)

    # 3) 保证区间包住点预测（两者都有限时才做）
    m_lp = np.isfinite(low) & np.isfinite(pred)
    low[m_lp] = np.minimum(low[m_lp], pred[m_lp])

    m_up = np.isfinite(up) & np.isfinite(pred)
    up[m_up] = np.maximum(up[m_up], pred[m_up])

    df[p], df[l], df[u] = pred, low, up
    return True


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Cannot find folder: {ROOT.resolve()}")

    # 只处理 results/DLinear_* 文件夹
    dlinear_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("DLinear_")])
    if not dlinear_dirs:
        print(f"No DLinear_* folders found under: {ROOT.resolve()}")
        return

    targets = []
    for d in dlinear_dirs:
        targets.extend(d.rglob("rolling_pred_step*.csv"))

    if not targets:
        print(f"No rolling_pred_step*.csv found under DLinear_* folders in: {ROOT.resolve()}")
        return

    print(f"Found {len(dlinear_dirs)} DLinear folders, {len(targets)} csv files to process.")

    for fp in sorted(targets):
        m = re.search(r"rolling_pred_step(\d+)\.csv$", fp.name)
        if not m:
            continue
        step = int(m.group(1))

        df = pd.read_csv(fp)
        ok = clip_one_step(df, step)
        if not ok:
            print(f"[SKIP] missing pred/lower/upper cols for step{step}: {fp}")
            continue

        # 备份一次
        bak = fp.with_suffix(fp.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(fp, bak)

        df.to_csv(fp, index=False)
        print(f"[OK] {fp}")

    print("Done.")


if __name__ == "__main__":
    main()