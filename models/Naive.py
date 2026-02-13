import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Naive (random-walk) baseline:
    predict all horizons as the last observed value.

    Output: [B, pred_len, C]
    """
    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len

        # Dummy parameter so optimizer/backward won't crash.
        # Does not change predictions because we add 0.0 * dummy in forward.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        last = x_enc[:, -1:, :]                  # [B, 1, C]
        out = last.repeat(1, self.pred_len, 1)   # [B, pred_len, C]
        out = out + 0.0 * self.dummy             # keep graph for backward, value unchanged
        return out

