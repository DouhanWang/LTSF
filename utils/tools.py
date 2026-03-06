import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
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

def visual(true, preds, path, lower=None, upper=None, seq_len=None,paper=False, show_xlabel=True, show_ylabel=True, show_xticklabels=True, show_yticklabels=True, ylim=None):
    """
    SAME calling as before.
    If true/preds/lower/upper are pandas Series with DatetimeIndex,
    x-axis will be exact dates; otherwise fallback to 0,1,2,...
    Produces line plot + shaded confidence interval.
    """

    # ---- helper to get x-axis ----
    def extract_x(x):
        if isinstance(x, pd.Series) and isinstance(x.index, pd.DatetimeIndex):
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
        if x is None:
            return None
        if isinstance(x, pd.Series):
            return x.values.astype(float)
        return np.asarray(x, dtype=float)

    true_vals = to_vals(true)
    pred_vals = to_vals(preds)
    lower_vals = to_vals(lower)
    upper_vals = to_vals(upper)

    n_true = len(true_vals)

    # ---- forecast start ----
    pred_start_idx = (n_true // 2) if seq_len is None else int(seq_len)
    pred_start_idx = max(1, min(pred_start_idx, n_true))  # safety

    # ---- x axis ----
    if x_dates is None:
        x_true = np.arange(n_true)
    else:
        x_true = np.asarray(x_dates)
        # try to regularize irregular datetime ticks (optional)
        try:
            x_dt = pd.to_datetime(x_true)
            if len(x_dt) >= 2:
                diffs = (x_dt[1:] - x_dt[:-1])
                step = pd.Series(diffs).mode().iloc[0]  # most common timedelta
                if not (diffs == step).all():
                    regular_range = pd.date_range(start=x_dt[0], periods=n_true, freq=step)
                    x_true = regular_range.to_pydatetime()
        except Exception:
            x_true = np.asarray(x_dates)

    # ---- align preds/bounds to future ----
    # Case A: pred_vals is same length as true (full series with future embedded)
    if len(pred_vals) == n_true:
        x_pred = x_true[pred_start_idx:]
        y_pred = pred_vals[pred_start_idx:]
        if lower_vals is not None and upper_vals is not None and len(lower_vals) == n_true and len(upper_vals) == n_true:
            lower_vals = lower_vals[pred_start_idx:]
            upper_vals = upper_vals[pred_start_idx:]
        else:
            # if bounds shape doesn't match, disable shading
            lower_vals, upper_vals = None, None
    else:
        # Case B: pred_vals is only the forecast horizon
        x_pred = x_true[pred_start_idx: pred_start_idx + len(pred_vals)]
        y_pred = pred_vals
        # bounds should match forecast horizon length if provided
        if lower_vals is not None and upper_vals is not None:
            if len(lower_vals) != len(y_pred) or len(upper_vals) != len(y_pred):
                lower_vals, upper_vals = None, None

    # ---- plotting ----
    plt.figure(figsize=(10, 6))
    ax = plt.gca()



    # --- y-limits (optional) ---
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    # --- paper mode: cleaner axes ---
    if paper:
        ax.tick_params(axis="both", labelsize=12)
        # fewer y ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        # fewer x ticks (dates)
        if x_dates is not None:
            locator = mdates.AutoDateLocator(minticks=3, maxticks=4)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


    # clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    axis_color = "#2B2B2B"
    for side in ["bottom", "left"]:
        ax.spines[side].set_linewidth(1.4)
        ax.spines[side].set_color(axis_color)

    ax.tick_params(axis="both", which="major", colors=axis_color, length=4, width=1.0, labelsize=18)
    # ground truth: line + small markers
    # ax.plot(
    #     x_true,
    #     true_vals,
    #     "-",
    #     color="black",
    #     linewidth=2.0,
    #     label="Ground Truth",
    #     zorder=3
    # )
    ax.scatter(
        x_true,
        true_vals,
        color="black",
        label="Ground Truth",
        s=16,
        zorder=4
    )

    # prediction: line + small markers (future only)
    pred_color = '#2E6F4E' # dark blue "#1F4E79"

    ax.plot(x_pred, y_pred, "-", color=pred_color, linewidth=2.2, zorder=4)


    # ax.scatter(
    #     x_pred,
    #     y_pred,
    #     color="#2ca02c",
    #     s=16,
    #     zorder=5
    # )

    # shaded confidence interval
    if lower_vals is not None and upper_vals is not None:
        # keep bounds consistent with mean
        lower_adj = np.minimum(lower_vals, y_pred)
        upper_adj = np.maximum(upper_vals, y_pred)

        mask = np.isfinite(lower_adj) & np.isfinite(upper_adj) & np.isfinite(y_pred)
        if mask.any():
            ax.fill_between(
                np.asarray(x_pred)[mask],
                lower_adj[mask],
                upper_adj[mask],
                color=pred_color,
                alpha=0.18,
                linewidth=0,
                zorder=2
            )

    # split line at last history date
    split_x = x_true[pred_start_idx - 1]
    split_color = "#8C8C8C"
    ax.axvline(
        split_x,
        color=split_color,
        linestyle=(0, (3, 3)),
        linewidth=1.0,
        alpha=0.9,
        zorder=1
    )

    ax.set_xlabel("Date" if x_dates is not None else "Time Step", fontsize=20)
    ax.set_ylabel("Incidence", fontsize=20)

    # de-duplicate legend labels (optional but helps)
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    # ax.legend(
    #     list(uniq.values()), list(uniq.keys()),
    #     frameon=False,
    #     loc="center left",
    #     bbox_to_anchor=(1.02, 0.5),  # ✅ legend to the right
    #     borderaxespad=0.0
    # )
    plt.tight_layout()

    # x tick formatting for dates
    if x_dates is not None:
        # Don't force ticks at every point (too crowded). Let matplotlib choose.
        plt.gcf().autofmt_xdate(rotation=30, ha="right")
    # --- axis labels / tick labels toggles ---
    if not show_xlabel:
        ax.set_xlabel("")
    if not show_ylabel:
        ax.set_ylabel("")

    if not show_xticklabels:
        ax.set_xticklabels([])
    if not show_yticklabels:
        ax.set_yticklabels([])
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