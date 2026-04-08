import os
import matplotlib as mpl
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


def visual(true, preds=None, name='./pic/test.pdf', lower=None, upper=None, seq_len=4,
           paper=True, show_xticklabels=True, show_yticklabels=True,
           show_xlabel=True, show_ylabel=True, ylim=None,
           pred_color='#ff7f0e', pred_label='Prediction'):
    """
    Results visualization
    """
    if paper:
        plt.rcParams.update({
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })
    else:
        plt.rc('font', family='serif')

    fig, ax = plt.subplots(figsize=(6, 4))
    
    len_t = len(true)
    x_true = np.arange(len_t)
    
    # Draw Ground Truth
    ax.plot(x_true, true, label='GroundTruth', color='black', marker='o', markersize=3.5, alpha=0.9, linestyle='None', zorder=50)


    ax.axvline(x=3, color='gray', linestyle=':', linewidth=1.5, alpha=0.8, zorder=49)

    # Draw Prediction
    if preds is not None:
        len_p = len(preds)
        x_pred = np.arange(len_t - len_p, len_t)
        
        ax.plot(x_pred, preds, label=pred_label, linewidth=1.5, color=pred_color, zorder=20)
        
        # Draw confidence intervals if provided
        if lower is not None and upper is not None:
            ax.fill_between(x_pred, lower, upper, color=pred_color, alpha=0.2, zorder=10)

    # Spines style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # Tick logic
    if show_xticklabels:
        ax.tick_params(axis='x', colors='#333333', bottom=True, labelbottom=True)
    else:
        ax.tick_params(axis='x', colors='#333333', bottom=True, labelbottom=False)

    if show_yticklabels:
        ax.tick_params(axis='y', colors='#333333', left=True, labelleft=True)
    else:
        ax.tick_params(axis='y', colors='#333333', left=True, labelleft=False)

    # Axis labels
    if show_xlabel:
        ax.set_xlabel('Date', fontsize=13, color='#333333')
    if show_ylabel:
        ax.set_ylabel('Incidence', fontsize=13, color='#333333')
        
    if ylim is not None:
        ax.set_ylim(ylim)

    # Force show Legend for every plot
    ax.legend(loc='best', frameon=True, fontsize=10, facecolor='white', framealpha=0.85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name, bbox_inches='tight', dpi=800)
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