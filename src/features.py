"""Feature construction. Every feature uses information available at the
close of day t only.

Two families:
  - HAR components (Corsi, 2009): daily, weekly and monthly realized
    volatility, built from the same zero-mean estimator as the target.
  - VIX: level at t, lags, and daily changes.

The monthly HAR window is 22 days while the forecast horizon is 21. This is
deliberate: 1/5/22 is Corsi's canonical specification, and a feature need not
share the target's window. Both are known at t, so neither leaks.
"""

import numpy as np
import pandas as pd

from .target import HORIZON, realized_vol

HAR_DAILY = 1
HAR_WEEKLY = 5
HAR_MONTHLY = 22

VIX_LAGS = (1, 5, 21)


def har_components(log_ret: pd.Series) -> pd.DataFrame:
    """Daily, weekly and monthly realized volatility, all known at t.

    Under the zero-mean convention the daily component reduces to
    |r(t)| * sqrt(252) * 100.
    """
    return pd.DataFrame(
        {
            "rv_d": realized_vol(log_ret, window=HAR_DAILY),
            "rv_w": realized_vol(log_ret, window=HAR_WEEKLY),
            "rv_m": realized_vol(log_ret, window=HAR_MONTHLY),
            # Realized volatility over exactly the horizon length, used by the
            # persistence baseline: "the next 21 days look like the last 21".
            "rv_h": realized_vol(log_ret, window=HORIZON),
        }
    )


def vix_features(vix: pd.Series, lags: tuple = VIX_LAGS) -> pd.DataFrame:
    """VIX level at t, selected lags, and the previous day's change.

    VIX(t) is the close of day t, so it is legitimately known at t and is
    named 'vix' rather than 'VIX_lag0'. Lags start at k = 1 by convention.
    """
    out = {"vix": vix}
    for k in lags:
        out[f"VIX_lag{k}"] = vix.shift(k)
    # Yesterday's VIX change: VIX(t-1) - VIX(t-2), known at t-1.
    out["DVIX_lag1"] = vix.diff(1).shift(1)
    return pd.DataFrame(out)


def nonlinear_vix_features(vix: pd.Series) -> pd.DataFrame:
    """Optional non-linear transforms of the VIX level at t."""
    return pd.DataFrame(
        {
            "vix_log": np.log(vix.clip(lower=1.0)),
            "vix_sq": vix**2,
        }
    )


def build_features(log_ret: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """Assemble the full feature matrix. All columns are known at the close of t."""
    return pd.concat(
        [har_components(log_ret), vix_features(vix), nonlinear_vix_features(vix)],
        axis=1,
    )


# Feature sets used by the models in models.py.
FEATURES_PERSISTENCE = ["rv_h"]
FEATURES_VIX = ["vix"]
FEATURES_HAR = ["rv_d", "rv_w", "rv_m"]
FEATURES_HAR_VIX = FEATURES_HAR + ["vix"]
FEATURES_ALL = FEATURES_HAR + [
    "rv_h",
    "vix",
    "VIX_lag1",
    "VIX_lag5",
    "VIX_lag21",
    "DVIX_lag1",
    "vix_log",
    "vix_sq",
]
