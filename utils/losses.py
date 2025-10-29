import torch


def quantile_loss(y_true, y_pred, quantiles):
    """
    y_true: [B, L, C]
    y_pred: [B, L, C * Q]
    quantiles: list or tensor of quantile levels (e.g. [0.1, 0.5, 0.9])
    """
    B, L, CQ = y_pred.shape
    C = y_true.shape[-1]
    Q = len(quantiles)

    # reshape to [B, L, C, Q]
    y_pred = y_pred.view(B, L, C, Q)

    losses = []
    for i, q in enumerate(quantiles):
        e = y_true - y_pred[..., i]
        loss_q = torch.max((q - 1) * e, q * e).unsqueeze(-1)
        losses.append(loss_q)

    losses = torch.cat(losses, dim=-1)
    # average over all dimensions
    return torch.mean(torch.sum(losses, dim=-1))
