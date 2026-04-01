# 检查两者 WIS 数组长度
import numpy as np

tabpfn_wis = np.load("./results/TabPFN_ts_real_Czechia_ILI/wis80_point_step4.npy", allow_pickle=True)
respicast_wis = np.load("./results/RespiCast_real_Czechia_ILI/wis80_point_step4.npy", allow_pickle=True)
Naive_wis = np.load("./results/Naive_real_Czechia_ILI/wis80_point_step4.npy", allow_pickle=True)
print("TabPFN WIS array length:", len(tabpfn_wis))
print("RespiCast WIS array length:", len(respicast_wis))
# print(tabpfn_wis)
# print(respicast_wis)

# print("TabPFN mean WIS:", np.nanmean(tabpfn_wis))
# print("RespiCast mean WIS:", np.nanmean(respicast_wis))