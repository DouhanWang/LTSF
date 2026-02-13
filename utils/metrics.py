import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean()


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))
def PICP(pred_lower, pred_upper, true):
    """Prediction Interval Coverage Probability"""
    covered = (true >= pred_lower) & (true <= pred_upper)
    return np.mean(covered)

def PINAW(pred_lower, pred_upper, true):
    """Prediction Interval Normalized Average Width"""
    return np.mean(pred_upper - pred_lower) / (np.max(true) - np.min(true))
def WMAPE(pred, true):
    """Weighted MAPE: sum|err| / sum|true|"""
    num = np.sum(np.abs(true - pred))
    den = np.sum(np.abs(true))
    return num / den if den != 0 else np.nan

def mean_WIS_interval(pred_lower, pred_upper, true, alpha):
    """
    pred_lower, pred_upper, true: arrays with same shape (T,) or (T,C)
    alpha: e.g. 0.05 for 95% interval
    """
    width = pred_upper - pred_lower
    below = (true < pred_lower) * (pred_lower - true)
    above = (true > pred_upper) * (true - pred_upper)
    is_alpha = width + (2.0 / alpha) * below + (2.0 / alpha) * above
    return np.mean(is_alpha)

# def metric(pred, true):
#     mae = MAE(pred, true)
#     mse = MSE(pred, true)
#     rmse = RMSE(pred, true)
#     mape = MAPE(pred, true)
#     mspe = MSPE(pred, true)
#     rse = RSE(pred, true)
#     corr = CORR(pred, true)
#
#     return mae, mse, rmse, mape, mspe, rse, corr
def metric(pred, true, quantiles=None):
    """
    pred: [B, L, C] for deterministic, or [B, L, C*Q] for quantile outputs
    true: [B, L, C]
    quantiles: list of quantile levels, e.g. [0.1, 0.5, 0.9]
    """

    if quantiles is not None and pred.shape[-1] != true.shape[-1]:
        # --- THIS IS THE FIX ---
        # We have quantile predictions that are "flat"

        # 1. Get dimensions
        num_quantiles = len(quantiles)  # e.g., 3
        num_features = true.shape[-1]  # e.g., 7

        # 2. Reshape pred from (B, L, C*Q) to (B, L, C, Q)
        # e.g., (190, 4, 21) -> (190, 4, 7, 3)
        try:
            pred_reshaped = pred.reshape(pred.shape[0], pred.shape[1], num_features, num_quantiles)
        except ValueError as e:
            print(f"Error reshaping pred: {e}")
            print(f"Pred shape: {pred.shape}, True shape: {true.shape}, Num quantiles: {num_quantiles}")
            # Fallback to just comparing first channel if reshape fails
            pred_median = pred[..., 0]

            # 3. Find median index (e.g., index 1 for [0.1, 0.5, 0.9])
        try:
            median_idx = quantiles.index(0.5)
        except ValueError:
            median_idx = num_quantiles // 2  # Fallback

        # 4. Extract median prediction
        pred_median = pred_reshaped[..., median_idx]

        # 5. Extract bounds for PICP/PINAW (using your new logic)
        lower_bound = pred_reshaped[..., 0]  # Assumes 0.1 is first
        upper_bound = pred_reshaped[..., -1]  # Assumes 0.9 is last

    else:
        # Standard deterministic prediction
        pred_median = pred
        pred_reshaped = None  # No quantiles

    # --- Calculate standard metrics on the median ---
    mae = MAE(pred_median, true)
    mse = MSE(pred_median, true)
    rmse = RMSE(pred_median, true)
    mape = MAPE(pred_median, true)
    mspe = MSPE(pred_median, true)
    rse = RSE(pred_median, true)
    wmape = WMAPE(pred_median, true)
    results = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'MSPE': mspe,
        'RSE': rse,
        'WMAPE': wmape,
    }

    # # --- Add coverage metrics if quantiles exist ---
    # if pred_reshaped is not None:
    #     picp = PICP(lower_bound, upper_bound, true)
    #     pinaw = PINAW(lower_bound, upper_bound, true)
    #     results.update({'PICP': picp, 'PINAW': pinaw})
    #     wis = mean_WIS(pred_reshaped, true, quantiles)
    #
    #     results.update({
    #         'PICP': picp,
    #         'PINAW': pinaw,
    #         'WIS': wis,  # mean WIS
    #     })

    return results
