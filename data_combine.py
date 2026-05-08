# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import pandas as pd
'''
This code combines the true data and the simulated (exogeneous) data into a single file (combined_Romania_ILI.csv). The real data is assigned an item_id of 0,
 while the simulated data's item_id is adjusted to start from 1. The combined data is then sorted and saved as a new CSV file.
'''

real_path = "./dataset/real_Romania_ILI.csv"          
sim_path = "./dataset/simulated_Romania_ILI.csv"
out_path = "./dataset/combined_Romania_ILI.csv"     

real = pd.read_csv(real_path)
sim = pd.read_csv(sim_path)


cols = ["item_id", "season_id", "anno", "settimana", "incidenza"]

# real data：set item_id=0
real["item_id"] = 0
real = real[cols]

# simulated：item_id from 0-999 -> 1-1000
sim["item_id"] = sim["item_id"].astype(int) + 1
sim = sim[cols]


combined = pd.concat([real, sim], ignore_index=True)


combined = combined.sort_values(["item_id", "season_id", "anno", "settimana"]).reset_index(drop=True)


combined.to_csv(out_path, index=False)
print("Saved:", out_path, "shape=", combined.shape)
