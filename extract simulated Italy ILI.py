import pandas as pd
in_path = "./dataset/simulated_Italy_ILI.csv"
out_path = "./dataset/simulated_Italy_ILI_item0.csv"

df = pd.read_csv(in_path)

# 只保留 item_id=0
df0 = df[df["item_id"] == 0].copy()


# 按时间排序（如果有 anno/settimana）
if "anno" in df0.columns and "settimana" in df0.columns:
    df0["settimana"] = df0["settimana"].astype(int)
    df0 = df0.sort_values(["anno", "settimana"]).reset_index(drop=True)

df0.to_csv(out_path, index=False)
