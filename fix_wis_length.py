import os
import re
import numpy as np
from pathlib import Path

# 【配置】根目录路径
ROOT = Path("./results")

# ==========================================
# 统一长度截取规则：键为 step，值为需要保留的“后 N 个”
# ==========================================
TARGET_LENGTHS = {
    1: 21, 
    2: 20, 
    3: 19, 
    4: 18
}

def process_all_wis_npy():
    print("🚀 --- 开始统一处理 wis80_point npy 文件 ---")
    stats = {"processed": 0, "skipped_perfect": 0, "error": 0}
    
    # 匹配文件名提取 step (1, 2, 3, 4)
    step_re = re.compile(r"wis80_point_step(\d+)\.npy", re.IGNORECASE)
    
    for p in ROOT.rglob("wis80_point_step*.npy"):
        path_str = str(p).lower()
        
        match = step_re.search(p.name)
        if not match:
            continue
        step = int(match.group(1))
        
        # 2. 获取当前 step 的目标截取长度
        target_len = TARGET_LENGTHS.get(step, 21)
        
        # 3. 读取、截断与保存
        try:
            arr = np.load(p)
            current_len = len(arr)
            
            # 情况 A: 原长度大于目标长度 -> 需要从后往前截取
            if current_len > target_len:
                new_arr = arr[-target_len:]  # 核心：取后 target_len 个
                np.save(p, new_arr)
                stats["processed"] += 1
                print(f"✅ [截断成功] {p.parent.name}/{p.name} | 原长 {current_len} -> 保留后 {target_len} 个")
                
            # 情况 B: 原长度刚好等于目标长度 -> 无需修改，节省磁盘读写
            elif current_len == target_len:
                stats["skipped_perfect"] += 1
                # print(f"ℹ️ [无需修改] {p.parent.name}/{p.name} | 长度已经是完美的 {target_len}")
                
            # 情况 C: 原长度居然比目标还要短 -> 报错警告
            else:
                stats["error"] += 1
                print(f"⚠️ [长度不足] {p.parent.name}/{p.name} | 需要后 {target_len} 个，但总共只有 {current_len} 个！")
                
        except Exception as e:
            stats["error"] += 1
            print(f"❌ [读取失败] {p}: {e}")

    print("\n🎉 === 统一处理任务完成 === ")
    print(f"🔄 成功截断并覆盖的文件数: {stats['processed']}")
    print(f"✅ 原本长度就已完美的文件数: {stats['skipped_perfect']}")
    
    if stats["error"] > 0:
        print(f"⚠️ 警告: 有 {stats['error']} 个文件报错或长度不足，请往上翻看终端输出。")

if __name__ == "__main__":
    process_all_wis_npy()