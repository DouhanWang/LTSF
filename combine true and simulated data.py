import pandas as pd

# ========= 1) 把单国 IT 导出成 simulated_Italy_ILI.csv，且 item_id=0 =========
real_path = "./dataset/per_country_csv/FR.csv"          # 你导出的IT文件
sim_path = "./dataset/simulated_France_ILI.csv"
out_path = "./dataset/combined_France_ILI.csv"     # 你想要的输出名

real = pd.read_csv(real_path)
sim = pd.read_csv(sim_path)

# --- 统一列 & 格式 ---
cols = ["item_id", "season_id", "anno", "settimana", "incidenza"]

# 真实数据：强制 item_id=0
real["item_id"] = 0
real = real[cols]

# simulated：item_id 从 0-999 -> 1-1000（整体+1，给真实数据让位）
sim["item_id"] = sim["item_id"].astype(int) + 1
sim = sim[cols]

# --- 合并 ---
combined = pd.concat([real, sim], ignore_index=True)

# （可选）排序：按 item_id、season_id、anno、settimana
combined = combined.sort_values(["item_id", "season_id", "anno", "settimana"]).reset_index(drop=True)

# --- 保存 ---
combined.to_csv(out_path, index=False)
print("Saved:", out_path, "shape=", combined.shape)
