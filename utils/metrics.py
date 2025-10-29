import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


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
    pred: [B, L, C] for deterministic, or [B, L, C, Q] for quantile outputs
    true: [B, L, C]
    quantiles: list of quantile levels, e.g. [0.1, 0.5, 0.9]
    """

    # If quantile predictions are used, select median (0.5)
    if pred.ndim == 4 and quantiles is not None:
        q_idx = quantiles.index(0.5)
        pred_median = pred[..., q_idx]
    else:
        pred_median = pred

    mae = MAE(pred_median, true)
    mse = MSE(pred_median, true)
    rmse = RMSE(pred_median, true)
    mape = MAPE(pred_median, true)
    mspe = MSPE(pred_median, true)
    rse = RSE(pred_median, true)
    corr = CORR(pred_median, true)

    results = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'MSPE': mspe,
        'RSE': rse,
        'CORR': corr,
    }

    # Add coverage metrics if quantile predictions exist
    if pred.ndim == 4 and quantiles is not None and len(quantiles) >= 3:
        lower = pred[..., 0]   # e.g. q=0.1
        upper = pred[..., -1]  # e.g. q=0.9
        picp = PICP(lower, upper, true)
        pinaw = PINAW(lower, upper, true)
        results.update({'PICP': picp, 'PINAW': pinaw})

    return results
