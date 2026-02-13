import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

import time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf # change previous Inf to inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# def visual(true, preds=None, name='./pic/test.pdf'):
#     """
#     Results visualization
#     """
#     plt.figure()
#     plt.plot(true, label='GroundTruth', linewidth=2)
#     if preds is not None:
#         plt.plot(preds, label='Prediction', linewidth=2)
#     plt.legend()
#     plt.savefig(name, bbox_inches='tight')


# def visual(true, preds, path, lower=None, upper=None, seq_len=None):
#     plt.figure(figsize=(10, 6))
#
#     # --- Find the prediction start index ---
#     if seq_len is None:
#         # Fallback if seq_len is not provided
#         pred_start_idx = len(true) // 2
#     else:
#         pred_start_idx = seq_len  # This is the correct history length (e.g., 4)
#
#     # --- Plot the main lines ---
#     plt.plot(true, label='GroundTruth', color='black')
#     plt.plot(preds, label='Prediction (Median)', color='blue')
#
#     # --- Plot the uncertainty band ---
#     if lower is not None and upper is not None:
#         plt.fill_between(
#             x=range(pred_start_idx, len(true)),  # X-axis range (e.g., from 4 to 8)
#             y1=lower[pred_start_idx:],  # Lower bound
#             y2=upper[pred_start_idx:],  # Upper bound
#             color='lightblue',
#             alpha=0.5,
#             label='95% Prediction Interval'
#         )
#
#
#     plt.axvline(x=pred_start_idx - 1, color='red', linestyle='--', label='Forecast Start')
#
#     # --- 2. Add Axis Labels ---
#     plt.xlabel('Time Step')
#     plt.ylabel('Incidenza')
#
#     plt.legend()
#     plt.savefig(path)
#     plt.close()

def visual(true, preds, path, lower=None, upper=None, seq_len=None):
    """
    SAME calling as before.
    If true/preds/lower/upper are pandas Series with DatetimeIndex,
    x-axis will be exact dates; otherwise fallback to 0,1,2,...
    Produces dots + per-dot uncertainty bars.
    """

    # ---- helper to get x-axis ----
    def extract_x(x):
        if isinstance(x, pd.Series):
            if isinstance(x.index, pd.DatetimeIndex):
                return x.index.to_pydatetime()
        return None

    x_dates = extract_x(true)
    if x_dates is None:
        x_dates = extract_x(preds)
    if x_dates is None and lower is not None:
        x_dates = extract_x(lower)
    if x_dates is None and upper is not None:
        x_dates = extract_x(upper)

    # ---- values ----
    def to_vals(x):
        if isinstance(x, pd.Series):
            return x.values.astype(float)
        return np.asarray(x, dtype=float)

    true_vals = to_vals(true)
    pred_vals = to_vals(preds)
    lower_vals = to_vals(lower) if lower is not None else None
    upper_vals = to_vals(upper) if upper is not None else None

    n_true = len(true_vals)

    # ---- forecast start ----
    pred_start_idx = (n_true // 2) if seq_len is None else seq_len

    # ---- x axis ----
    if x_dates is None:
        x_true = np.arange(n_true)
    else:
        x_true = np.asarray(x_dates)
        try:
            x_dt = pd.to_datetime(x_true)
            if len(x_dt) >= 2:
                diffs = (x_dt[1:] - x_dt[:-1])
                step = pd.Series(diffs).mode().iloc[0]   # most common timedelta
                if not (diffs == step).all():
                    regular_range = pd.date_range(
                        start=x_dt[0], periods=n_true, freq=step
                    )
                    x_true = regular_range.to_pydatetime()
        except Exception:
            # if anything goes wrong, just fall back to the raw dates
            x_true = np.asarray(x_dates)
    # ---- align preds/bounds to future ----
    if len(pred_vals) == n_true:
        x_pred = x_true[pred_start_idx:]
        y_pred = pred_vals[pred_start_idx:]
        if lower_vals is not None and upper_vals is not None:
            lower_vals = lower_vals[pred_start_idx:]
            upper_vals = upper_vals[pred_start_idx:]
    else:
        x_pred = x_true[pred_start_idx: pred_start_idx + len(pred_vals)]
        y_pred = pred_vals

    # ---- plot ----
    plt.figure(figsize=(10, 6))

    # history truth as dots
    plt.plot(
        x_true, #[:pred_start_idx]
        true_vals, #[:pred_start_idx]
        "o",
        color="blue",
        label="Ground Truth",
    )

    # predictions as dots
    plt.scatter(
        x_pred,
        y_pred,
        color="red",
        label="Prediction",
    )

    # uncertainty bars
    if lower_vals is not None and upper_vals is not None:
        # ensure bounds are consistent with the mean (no crossing)
        lower_vals = np.minimum(lower_vals, y_pred)
        upper_vals = np.maximum(upper_vals, y_pred)

        yerr_low = y_pred - lower_vals
        yerr_high = upper_vals - y_pred

        # numeric safety: clip to non-negative
        yerr_low = np.clip(yerr_low, 0, None)
        yerr_high = np.clip(yerr_high, 0, None)

        yerr = [yerr_low, yerr_high]

        plt.errorbar(
            x_pred,
            y_pred,
            yerr=yerr,
            fmt="none",
            alpha=0.3,
            color="red",
            label="Prediction Interval",
        )

    # split line at last history date
    split_x = x_true[pred_start_idx - 1]
    plt.axvline(split_x, color="red", linestyle="--", label="Prediction Split")

    # plt.title("Italy ILI Incidence Forecast (dot plot with uncertainty)")
    plt.xlabel("Date" if x_dates is not None else "Time Step")
    plt.ylabel("Incidenza")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # if x_dates is not None:
    #     plt.gcf().autofmt_xdate()
    if x_dates is not None:
        ax = plt.gca()
        ax.set_xticks(x_true)  # put ticks exactly at your data dates
        ax.set_xticklabels(
            [pd.to_datetime(d).strftime("%Y-%m-%d") for d in x_true],
            rotation=45, ha="right"
        )

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))