# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import os
import re
import numpy as np
from pathlib import Path

# 【配置】根目录路径
ROOT = Path("./results")


TARGET_LENGTHS = {
    1: 21, 
    2: 20, 
    3: 19, 
    4: 18
}

def process_all_wis_npy():
    print("🚀 --- Step 1: Start processing wis80_point npy files ---")
    stats = {"processed": 0, "skipped_perfect": 0, "error": 0}
    
    # find filename with step (1, 2, 3, 4)
    step_re = re.compile(r"wis80_point_step(\d+)\.npy", re.IGNORECASE)
    
    for p in ROOT.rglob("wis80_point_step*.npy"):
        path_str = str(p).lower()
        
        match = step_re.search(p.name)
        if not match:
            continue
        step = int(match.group(1))
        
        # 2. get target length for current step
        target_len = TARGET_LENGTHS.get(step, 21)
        
        # 3. read, truncate, and save
        try:
            arr = np.load(p)
            current_len = len(arr)
            
            # case A: raw length greater than target length -> need to truncate from the end
            if current_len > target_len:
                new_arr = arr[-target_len:]  # core: take the last target_len elements
                np.save(p, new_arr)
                stats["processed"] += 1
                print(f"✅ [Truncation Successful] {p.parent.name}/{p.name} | Original length {current_len} -> Retained {target_len} elements")
                
            # case B: raw length exactly equals target length -> no modification needed, save disk I/O
            elif current_len == target_len:
                stats["skipped_perfect"] += 1
                # print(f"ℹ️ [No Modification Needed] {p.parent.name}/{p.name} | Length is already perfect {target_len}")
                
            # case C: raw length is shorter than target -> error/warning
            else:
                stats["error"] += 1
                print(f"⚠️ [Insufficient Length] {p.parent.name}/{p.name} | Need {target_len} elements, but only {current_len} available!")
                
        except Exception as e:
            stats["error"] += 1
            print(f"❌ [Read Failed] {p}: {e}")

    print("\n🎉 === Uniform Processing Task Completed === ")
    print(f"🔄 Successfully Truncated and Overwritten Files: {stats['processed']}")
    print(f"✅ Files with Perfect Length: {stats['skipped_perfect']}")
    
    if stats["error"] > 0:
        print(f"⚠️ Warning: There are {stats['error']} files with errors or insufficient length. Please scroll up to view the terminal output.")

if __name__ == "__main__":
    process_all_wis_npy()