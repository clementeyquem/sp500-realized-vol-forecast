"""Realized volatility and the forward-looking prediction target.

Convention (fixed, see CLAUDE.md section 3.1): realized volatility is the
zero-mean estimator, i.e. the square root of the mean squared log return over
the window, annualized by sqrt(252) and expressed in volatility points.

    rv(t) = sqrt( mean( r(t-W+1)^2 .. r(t)^2 ) ) * sqrt(252) * 100

This is the standard realized-variance estimator. It is NOT pandas' .std(),
which subtracts the window mean and uses ddof=1. The two differ by up to
3.84 volatility points on 2000-2024 data, which matters because QLIKE is
defined on variance.
"""

import numpy as np
import pandas as pd

HORIZON = 21  # trading days; the forecast horizon H
ANNUALIZATION = 252


def realized_vol(log_ret: pd.Series, window: int = HORIZON) -> pd.Series:
    """Backward-looking annualized realized volatility, in volatility points.

    rv(t) uses returns r(t-window+1) .. r(t), so it is known at the close of t.
    """
    mean_sq = (log_ret**2).rolling(window).mean()
    return np.sqrt(mean_sq) * np.sqrt(ANNUALIZATION) * 100


def forward_realized_vol(log_ret: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Forward-looking target: realized volatility over r(t+1) .. r(t+horizon).

    Built by shifting the backward-looking estimator by -horizon:
    realized_vol(t+horizon) covers exactly r(t+1) .. r(t+horizon), so no
    information dated t or earlier enters the target. The last `horizon` rows
    are NaN and must be dropped, never imputed.
    """
    return realized_vol(log_ret, window=horizon).shift(-horizon)
