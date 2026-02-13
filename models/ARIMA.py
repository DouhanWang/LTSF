import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Robust ARIMA baseline for rolling forecasting.

    - Fits per forward call (per window / per sample).
    - Returns point forecast [B, pred_len, C]
    - Stores 80% CI in:
        self.last_lower, self.last_upper   (numpy arrays, shape [B, pred_len, C])
    """
    def __init__(self, args):
        super().__init__()
        self.pred_len = int(args.pred_len)

        # ARIMA order from args
        self.p = int(getattr(args, "arima_p", 1))
        self.d = int(getattr(args, "arima_d", 0))
        self.q = int(getattr(args, "arima_q", 0))

        # 80% CI -> alpha=0.2
        self.alpha = float(getattr(args, "arima_alpha", 0.2))

        # Optional: add drift/trend (recommended when d=1)
        # For statsmodels ARIMA:
        #   trend="n" (none), "c" (constant), "t" (linear), "ct" (both)
        self.trend = str(getattr(args, "arima_trend", "n"))
        if self.d == 1 and self.trend == "n":
            # very common to use drift with d=1
            self.trend = "t"

        # Fit controls
        self.maxiter = int(getattr(args, "arima_maxiter", 200))

        # fallback orders to try if the requested order fails
        # keep small/simple to avoid convergence issues
        self.fallback_orders = [
            (self.p, self.d, self.q),
            (0, 1, 1),
            (1, 1, 0),
            (0, 1, 0),  # random walk
            (1, 0, 0),
        ]

        # dummy param so optimizer/backward in your pipeline won't crash
        self.dummy = nn.Parameter(torch.zeros(1))

        self.last_lower = None
        self.last_upper = None

    @staticmethod
    def _safe_last_value(y: np.ndarray) -> float:
        y = y[np.isfinite(y)]
        return float(y[-1]) if y.size > 0 else 0.0

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        x_enc: [B, L, C] (scaled or real, depends on pipeline)
        """
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

                # clean
                y = y[np.isfinite(y)]
                last = self._safe_last_value(y)

                # if too short / constant -> naive
                if y.size < 8 or np.allclose(y, y[-1]):
                    preds[b, :, c] = last
                    lowers[b, :, c] = last
                    uppers[b, :, c] = last
                    continue

                # Try a few orders (requested first, then fallbacks)
                fitted = False
                for (pp, dd, qq) in self.fallback_orders:
                    # need enough points
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
                        res = model.fit(method_kwargs={"maxiter": self.maxiter})

                        fc = res.get_forecast(steps=self.pred_len)
                        mean = np.asarray(fc.predicted_mean, dtype=np.float64)

                        # CI: sometimes conf_int can fail; fallback to normal approx
                        try:
                            ci = fc.conf_int(alpha=self.alpha)
                            lo = np.asarray(ci[:, 0], dtype=np.float64)
                            hi = np.asarray(ci[:, 1], dtype=np.float64)
                        except Exception:
                            # normal approx using forecast variance if available
                            try:
                                var = np.asarray(fc.var_pred_mean, dtype=np.float64)
                                se = np.sqrt(np.maximum(var, 1e-12))
                                # 80% -> z ≈ 1.2816
                                z = 1.2815515655446004
                                lo = mean - z * se
                                hi = mean + z * se
                            except Exception:
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
                    # ultimate fallback: last value
                    preds[b, :, c] = last
                    lowers[b, :, c] = last
                    uppers[b, :, c] = last

        self.last_lower = lowers
        self.last_upper = uppers

        out = torch.from_numpy(preds).to(device)
        out = out + 0.0 * self.dummy
        return out
