# Copyright 2026 DouhanWang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import warnings

import numpy as np
import torch
import torch.nn as nn
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
        self.last_orders: numpy array [B, C, 3] with the fitted (p, d, q)
    """

    def __init__(self, args):
        super().__init__()
        self.pred_len = int(args.pred_len)

        self.p = int(getattr(args, "arima_p", 1))
        self.d = int(getattr(args, "arima_d", 1))
        self.q = int(getattr(args, "arima_q", 0))

        self.alpha = float(getattr(args, "arima_alpha", 0.2))
        self.trend = str(getattr(args, "arima_trend", "n"))
        self.maxiter = int(getattr(args, "arima_maxiter", 200))

        self.auto = bool(getattr(args, "arima_auto", False))
        self.start_p = int(getattr(args, "arima_start_p", 2))
        self.start_q = int(getattr(args, "arima_start_q", 2))
        self.min_p = int(getattr(args, "arima_min_p", 0))
        self.max_p = int(getattr(args, "arima_max_p", 5))
        self.min_d = int(getattr(args, "arima_min_d", 0))
        self.max_d = int(getattr(args, "arima_max_d", 2))
        self.min_q = int(getattr(args, "arima_min_q", 0))
        self.max_q = int(getattr(args, "arima_max_q", 5))
        self.ic = str(getattr(args, "arima_ic", "aic")).lower()
        self.test = str(getattr(args, "arima_test", "kpss")).lower()
        self.max_order = int(getattr(args, "arima_max_order", 5))

        # dummy param so optimizer/backward won't crash
        self.dummy = nn.Parameter(torch.zeros(1))

        self.last_lower = None
        self.last_upper = None
        self.last_orders = None

        if self.d == 1 and self.trend == "n":
            self.trend = "t"

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

    def _trend_for_order(self, d: int) -> str:
        if self.trend == "n" and d == 1:
            return "t"
        return self.trend

    def _fit_statsmodels(self, y: np.ndarray, order):
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(
            y,
            order=order,
            trend=self._trend_for_order(order[1]),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            return model.fit(method_kwargs={"maxiter": self.maxiter})

    def _valid_auto_orders(self, y_size: int):
        min_p = max(0, self.min_p)
        max_p = max(min_p, self.max_p)
        min_d = max(0, self.min_d)
        max_d = max(min_d, self.max_d)
        min_q = max(0, self.min_q)
        max_q = max(min_q, self.max_q)

        for dd in range(min_d, max_d + 1):
            for pp in range(min_p, max_p + 1):
                for qq in range(min_q, max_q + 1):
                    if pp == 0 and dd == 0 and qq == 0:
                        continue
                    if self.max_order is not None and pp + qq > self.max_order:
                        continue
                    if y_size < max(10, pp + qq + dd + 5):
                        continue
                    yield (pp, dd, qq)

    def _select_order_with_pmdarima(self, y: np.ndarray):
        try:
            import pmdarima as pm
        except Exception:
            return None

        start_p = min(max(0, self.start_p), max(self.min_p, self.max_p))
        start_q = min(max(0, self.start_q), max(self.min_q, self.max_q))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = pm.auto_arima(
                    y,
                    start_p=start_p,
                    max_p=max(self.min_p, self.max_p),
                    start_q=start_q,
                    max_q=max(self.min_q, self.max_q),
                    d=None,
                    max_d=max(self.min_d, self.max_d),
                    max_order=self.max_order,
                    seasonal=False,
                    information_criterion=self.ic,
                    test=self.test,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False,
                )
            order = tuple(int(v) for v in model.order)
            if (
                self.min_p <= order[0] <= self.max_p
                and self.min_d <= order[1] <= self.max_d
                and self.min_q <= order[2] <= self.max_q
            ):
                return order
        except Exception:
            return None
        return None

    def _select_order_by_grid(self, y: np.ndarray):
        best_order = None
        best_score = np.inf

        for order in self._valid_auto_orders(y.size):
            try:
                res = self._fit_statsmodels(y, order)
                score = getattr(res, self.ic, np.inf)
                if np.isfinite(score) and score < best_score:
                    best_score = float(score)
                    best_order = order
            except Exception:
                continue

        return best_order

    def _orders_for_series(self, y: np.ndarray):
        fallback_orders = self.try_orders
        if self.auto:
            if self.min_q > 0:
                fallback_orders = [order for order in self.try_orders if order[2] >= self.min_q]
            order = self._select_order_with_pmdarima(y)
            if order is None:
                order = self._select_order_by_grid(y)
            if order is not None:
                return [order] + [o for o in fallback_orders if o != order]

        return fallback_orders

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        device = x_enc.device
        x_np = x_enc.detach().cpu().numpy()
        B, L, C = x_np.shape

        preds = np.zeros((B, self.pred_len, C), dtype=np.float32)
        lowers = np.zeros((B, self.pred_len, C), dtype=np.float32)
        uppers = np.zeros((B, self.pred_len, C), dtype=np.float32)
        orders = np.full((B, C, 3), -1, dtype=np.int32)

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

                for (pp, dd, qq) in self._orders_for_series(y):
                    if y.size < max(10, pp + qq + dd + 5):
                        continue
                    try:
                        res = self._fit_statsmodels(y, (pp, dd, qq))
                        fc = res.get_forecast(steps=self.pred_len)
                        mean = np.asarray(fc.predicted_mean, dtype=np.float64)

                        try:
                            ci = fc.conf_int(alpha=self.alpha)  # [pred_len, 2]
                            lo = np.asarray(ci[:, 0], dtype=np.float64)
                            hi = np.asarray(ci[:, 1], dtype=np.float64)
                        except Exception:
                            lo = mean.copy()
                            hi = mean.copy()

                        preds[b, :, c] = mean.astype(np.float32)
                        lowers[b, :, c] = lo.astype(np.float32)
                        uppers[b, :, c] = hi.astype(np.float32)
                        orders[b, c] = (pp, dd, qq)

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
        self.last_orders = orders

        out = torch.from_numpy(preds).to(device)
        out = out + 0.0 * self.dummy
        return out
