import pandas as pd
import numpy as np

from typing import Union

ArrayLike = Union[list, tuple, np.ndarray]

def median_ensemble(pred_tab: ArrayLike, pred_auto: ArrayLike, pred_dlinear: ArrayLike) -> np.ndarray:
    """
    Median ensemble across 3 model predictions.
    Accepts scalars, 1D arrays, or lists. Returns a NumPy array.
    """
    tab = np.asarray(pred_tab, dtype=float)
    auto = np.asarray(pred_auto, dtype=float)
    dlin = np.asarray(pred_dlinear, dtype=float)

    # Allow scalar inputs
    if tab.ndim == 0 and auto.ndim == 0 and dlin.ndim == 0:
        return np.array(float(np.median([tab.item(), auto.item(), dlin.item()])))

    # Ensure 1D shape
    tab = tab.reshape(-1)
    auto = auto.reshape(-1)
    dlin = dlin.reshape(-1)

    if not (tab.shape == auto.shape == dlin.shape):
        raise ValueError(f"Shape mismatch: tab={tab.shape}, auto={auto.shape}, dlinear={dlin.shape}")

    stacked = np.vstack([tab, auto, dlin])   # shape: (3, T)
    return np.median(stacked, axis=0)        # shape: (T,)

# -----------------------
if __name__ == "__main__":
    tab_in_path = "results/"
    dlin_in_path = "results/italy_ili_MS_uncertainty_4_4_augmentdata_shift_only_DLinear_custom_ftMS_sl4_ll2_pl4_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/pred.npy"
    auto_in_path = "results/italy_ili_MS_incidenza_sdscaler_uncertainty_6_4_augmentdata_shift_only_Autoformer_custom_ftMS_sl6_ll3_pl4_dm64_nh4_el2_dl1_df128_fc3_ebtimeF_dtTrue_Exp_0/pred.npy"

    #tab_preds = [10.2, 11.0, 12.3, 13.5]
    auto_preds = np.load(auto_in_path)
    dlin_preds = np.load(dlin_in_path)
    print(dlin_preds)
# ens_series = median_ensemble(auto_preds, dlin_preds)
# print("Ensemble series:", ens_series)
