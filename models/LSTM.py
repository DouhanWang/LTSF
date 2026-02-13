import torch
import torch.nn as nn


class Model(nn.Module):
    """
    LSTM forecaster (direct multi-step).
    - Input : x_enc [B, seq_len, enc_in]
    - Output: [B, pred_len, c_out]
    Forward signature aligns with your Exp_Main calling style:
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    We ignore time marks for a plain LSTM baseline.
    """

    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        self.enc_in = args.enc_in
        self.c_out = args.c_out

        hidden_size = args.d_model          # reuse existing arg
        num_layers = args.e_layers          # reuse existing arg
        dropout = args.dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Direct multi-step head: last hidden -> pred_len * c_out
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.pred_len * self.c_out),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        x_enc: [B, seq_len, enc_in]
        returns: [B, pred_len, c_out]
        """
        # Safety: ensure float
        if not torch.is_floating_point(x_enc):
            x_enc = x_enc.float()

        out, (h_n, c_n) = self.lstm(x_enc)      # h_n: [num_layers, B, hidden]
        h_last = h_n[-1]                        # [B, hidden]

        y = self.head(h_last)                   # [B, pred_len*c_out]
        y = y.view(y.shape[0], self.pred_len, self.c_out)
        return y
