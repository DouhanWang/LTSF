import numpy as np
import torch
import torch.nn as nn
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

class Model(nn.Module):
    """
    ARIMA baseline fitted per forward call.

    Inputs:
        x_enc: [B, L, C]
    Outputs:
        out:   [B, pred_len, C]  (point forecast)
    Side effects:
        self.last_lower, self.last_upper: numpy arrays [B, pred_len, C] for 80% CI
    """
    def __init__(self, args):
        super().__init__()
        self.pred_len = int(args.pred_len)

        # ARIMA order from args
        self.p = int(getattr(args, "arima_p", 1))
        self.d = int(getattr(args, "arima_d", 1))
        self.q = int(getattr(args, "arima_q", 0))

        # 80% CI -> alpha=0.2
        self.alpha = float(getattr(args, "arima_alpha", 0.2))

        # dummy param so optimizer/backward won't crash
        self.dummy = nn.Parameter(torch.zeros(1))

        self.last_lower = None
        self.last_upper = None

        self.trend = str(getattr(args, "arima_trend", "n"))
        self.maxiter = int(getattr(args, "arima_maxiter", 200))

        # 推荐：如果 d=1 且你没显式给 trend，就默认用 drift
        if self.d == 1 and self.trend == "n":
            self.trend = "t"
        # Fallback orders if fitting fails (keep small/simple)
        # Try requested first, then common robust alternatives
        self.try_orders = [
            (self.p, self.d, self.q),
            (1, 1, 0),
            (0, 1, 1),
            (0, 1, 0),  # random walk
            (1, 0, 0),
        ]

    @staticmethod
    def _last_value(y: np.ndarray) -> float:
        y = y[np.isfinite(y)]
        return float(y[-1]) if y.size > 0 else 0.0

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        device = x_enc.device
        x_np = x_enc.detach().cpu().numpy()
        B, L, C = x_np.shape

        preds = np.zeros((B, self.pred_len, C), dtype=np.float32)
        lowers = np.zeros((B, self.pred_len, C), dtype=np.float32)
        uppers = np.zeros((B, self.pred_len, C), dtype=np.float32)

        from statsmodels.tsa.arima.model import ARIMA

        for b in range(B):
            for c in range(C):
                y = x_np[b, :, c].astype(np.float64)
                y = y[np.isfinite(y)]
                last = self._last_value(y)

                # Too short or constant -> naive
                if y.size < 8 or np.allclose(y, y[-1]):
                    preds[b, :, c] = last
                    lowers[b, :, c] = last
                    uppers[b, :, c] = last
                    continue

                fitted = False

                for (pp, dd, qq) in self.try_orders:
                    # Need enough points for the chosen order
                    if y.size < max(10, pp + qq + dd + 5):
                        continue
                    try:
                        model = ARIMA(
                            y,
                            order=(pp, dd, qq),
                            trend=self.trend,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConvergenceWarning)
                            res = model.fit(method_kwargs={"maxiter": self.maxiter})

                        fc = res.get_forecast(steps=self.pred_len)
                        mean = np.asarray(fc.predicted_mean, dtype=np.float64)

                        # CI
                        try:
                            ci = fc.conf_int(alpha=self.alpha)  # [pred_len, 2]
                            lo = np.asarray(ci[:, 0], dtype=np.float64)
                            hi = np.asarray(ci[:, 1], dtype=np.float64)
                        except Exception:
                            # fallback: degenerate CI if conf_int fails
                            lo = mean.copy()
                            hi = mean.copy()

                        preds[b, :, c] = mean.astype(np.float32)
                        lowers[b, :, c] = lo.astype(np.float32)
                        uppers[b, :, c] = hi.astype(np.float32)

                        fitted = True
                        break

                    except Exception:
                        continue

                if not fitted:
                    preds[b, :, c] = last
                    lowers[b, :, c] = last
                    uppers[b, :, c] = last

        self.last_lower = lowers
        self.last_upper = uppers

        out = torch.from_numpy(preds).to(device)
        out = out + 0.0 * self.dummy
        return out