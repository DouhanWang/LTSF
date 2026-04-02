import os
import numpy as np
import pandas as pd

# ==========================================
# 统一长度截取规则
# ==========================================
TARGET_LENGTHS = {1: 21, 2: 20, 3: 19, 4: 18}

def wis_pointwise(pred_lower, pred_upper, true, alpha):
    """
    计算逐点的 WIS (Interval Score)
    """
    pred_lower = np.asarray(pred_lower, dtype=float)
    pred_upper = np.asarray(pred_upper, dtype=float)
    true = np.asarray(true, dtype=float)

    width = pred_upper - pred_lower
    below = (true < pred_lower) * (pred_lower - true)
    above = (true > pred_upper) * (true - pred_upper)
    is_alpha = width + (2.0 / alpha) * below + (2.0 / alpha) * above
    return is_alpha

def compute_and_save(country, step, model_name, alpha=0.2):
    target_dir = f"results/{model_name}_real_{country}_ILI"
    target_path = f"{target_dir}/rolling_pred_step{step}.csv"
    naive_path = f"results/Naive_real_{country}_ILI/rolling_pred_step{step}.csv"

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"缺失预测文件: {target_path}")
    if not os.path.exists(naive_path):
        raise FileNotFoundError(f"缺失 Naive 真实值文件: {naive_path}")

    df_target = pd.read_csv(target_path)
    df_naive = pd.read_csv(naive_path)

    if "date" not in df_target.columns or "date" not in df_naive.columns:
        raise ValueError(f"严重异常：文件缺失 date 列！")

    df_target["date_str"] = df_target["date"].astype(str).str.strip()
    df_naive["date_str"] = df_naive["date"].astype(str).str.strip()

    true_col = "TRUE" if "TRUE" in df_naive.columns else ("true" if "true" in df_naive.columns else None)
    if not true_col:
        raise ValueError(f"严重异常：Naive 文件中找不到 TRUE 列！")

    lower_col = next((c for c in df_target.columns if "lower" in c.lower() or "0.1" in c or c.lower().startswith("lo")), None)
    upper_col = next((c for c in df_target.columns if "upper" in c.lower() or "0.9" in c or c.lower().startswith("hi")), None)

    if not lower_col or not upper_col:
        raise ValueError(f"找不到 lower / upper 列！目前拥有的列: {list(df_target.columns)}")

    # 1. 完美左连接合并数据
    df_merged = pd.merge(
        df_target[["date_str", lower_col, upper_col]], 
        df_naive[["date_str", true_col]], 
        on="date_str", 
        how="left"
    )
    
    # 2. 【核心修改：一步到位强制裁切】
    # 在提取数据和计算任何指标前，直接把整个表格切到目标长度！
    target_len = TARGET_LENGTHS.get(step, 21)
    if len(df_merged) > target_len:
        df_merged = df_merged.tail(target_len).copy()

    # 3. 提取裁切后干净的数据
    lower = df_merged[lower_col].to_numpy(dtype=float)
    upper = df_merged[upper_col].to_numpy(dtype=float)
    true = df_merged[true_col].to_numpy(dtype=float)

    # 4. 过滤掉缺失值
    mask = np.isfinite(true) & np.isfinite(lower) & np.isfinite(upper)
    true_clean = true[mask]
    lower_clean = lower[mask]
    upper_clean = upper[mask]

    if true_clean.size == 0:
        raise ValueError(f"没有找到有效的 true/lower/upper 对齐数据点！(可能是日期没匹配上)")

    if np.any(upper_clean < lower_clean):
        idx = np.where(upper_clean < lower_clean)[0][:5]
        raise ValueError(f"严重错误：upper 小于 lower，出现在行 {idx}")

    # 5. 计算并覆盖保存
    wis_points = wis_pointwise(lower_clean, upper_clean, true_clean, alpha=alpha)
    
    os.makedirs(target_dir, exist_ok=True)
    out_path = f"{target_dir}/wis80_point_step{step}.npy"
    np.save(out_path, wis_points)

    mean_wis = float(np.mean(wis_points))
    picp = float(np.mean((true_clean >= lower_clean) & (true_clean <= upper_clean)))
    mean_width = float(np.mean(upper_clean - lower_clean))

    return out_path, mean_wis, picp, mean_width, len(wis_points)

def main():
    countries = ["Belgium", "Czechia", "Denmark", "France","Ireland","Italy","Netherlands","Poland","Romania"]
    alpha = 0.2  

    MODELS = ["Ensembleun"] #"Ensemble", "Respicast","TabPFN_ts"

    for model in MODELS:
        print(f"\n==========================================")
        print(f"🚀 开始计算模型: {model}")
        print(f"==========================================")
        
        for c in countries:
            print(f"\n=== {c} ===")
            for step in [1, 2, 3, 4]:
                try:
                    out_path, mean_wis, picp, width, n = compute_and_save(c, step, model, alpha=alpha)
                    print(f"step{step}: 成功生成 | meanWIS80={mean_wis:.4f} | PICP80={picp:.3f} | width={width:.3f} | 最终长度 n={n}")
                except Exception as e:
                    print(f"step{step}: ❌ [跳过] {e}")

if __name__ == "__main__":
    main()