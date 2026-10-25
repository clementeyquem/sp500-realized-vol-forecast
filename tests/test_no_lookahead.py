"""Leakage tests: features must not use information dated after t, and the
target must not use information dated t or earlier.

These are the executable form of the rules in CLAUDE.md sections 3.2 and 5.6.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.target import HORIZON, forward_realized_vol, realized_vol

SEED = 42
N = 400
SPLIT = 250  # the index t used as the "present" in the tests


@pytest.fixture
def series():
    """Synthetic but realistic daily returns and VIX levels."""
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2010-01-01", periods=N)
    log_ret = pd.Series(rng.normal(0, 0.01, N), index=idx)
    vix = pd.Series(15 + 5 * rng.random(N), index=idx)
    return log_ret, vix


def test_features_ignore_the_future(series):
    """Perturbing returns and VIX strictly after t must not change any
    feature value at or before t."""
    log_ret, vix = series
    base = build_features(log_ret, vix)

    rng = np.random.default_rng(SEED + 1)
    log_ret_p = log_ret.copy()
    vix_p = vix.copy()
    # Replace everything after t with wildly different values.
    log_ret_p.iloc[SPLIT + 1 :] = rng.normal(0, 0.10, N - SPLIT - 1)
    vix_p.iloc[SPLIT + 1 :] = 80.0

    perturbed = build_features(log_ret_p, vix_p)

    pd.testing.assert_frame_equal(
        base.iloc[: SPLIT + 1], perturbed.iloc[: SPLIT + 1]
    )


def test_perturbation_is_actually_visible(series):
    """Guard against a vacuous test: the perturbation must change something
    after t, otherwise test_features_ignore_the_future proves nothing."""
    log_ret, vix = series
    base = build_features(log_ret, vix)

    vix_p = vix.copy()
    vix_p.iloc[SPLIT + 1 :] = 80.0
    perturbed = build_features(log_ret, vix_p)

    assert not base.iloc[SPLIT + 1 :].equals(perturbed.iloc[SPLIT + 1 :])


def test_target_ignores_the_past(series):
    """The forward target at t must not depend on returns dated t or earlier."""
    log_ret, _ = series
    base = forward_realized_vol(log_ret)

    rng = np.random.default_rng(SEED + 2)
    log_ret_p = log_ret.copy()
    log_ret_p.iloc[: SPLIT + 1] = rng.normal(0, 0.10, SPLIT + 1)

    perturbed = forward_realized_vol(log_ret_p)

    assert base.iloc[SPLIT] == pytest.approx(perturbed.iloc[SPLIT], abs=1e-12)


def test_target_alignment_matches_manual_computation(series):
    """forward_realized_vol(t) must equal the realized vol of r(t+1)..r(t+H)."""
    log_ret, _ = series
    target = forward_realized_vol(log_ret)

    window = log_ret.iloc[SPLIT + 1 : SPLIT + 1 + HORIZON]
    manual = np.sqrt((window**2).mean()) * np.sqrt(252) * 100

    assert target.iloc[SPLIT] == pytest.approx(manual, abs=1e-12)
    assert len(window) == HORIZON


def test_target_tail_is_nan(series):
    """The last H observations have no future, so the target must be NaN."""
    log_ret, _ = series
    target = forward_realized_vol(log_ret)
    assert target.iloc[-HORIZON:].isna().all()
    assert target.iloc[: -HORIZON].notna().sum() > 0


def test_realized_vol_is_zero_mean_not_std(series):
    """Document the chosen convention: the estimator is sqrt(mean(r^2)),
    not pandas' .std(), and the two genuinely differ."""
    log_ret, _ = series
    chosen = realized_vol(log_ret, window=HORIZON)
    demeaned = log_ret.rolling(HORIZON).std() * np.sqrt(252) * 100

    assert not np.allclose(
        chosen.dropna().to_numpy(), demeaned.dropna().to_numpy()
    )

    # Exact relationship between the two estimators. With n = HORIZON,
    #   chosen^2   = (var_ddof0 + mean(r)^2) * k
    #   demeaned^2 = var_ddof0 * n / (n - 1) * k
    # hence ratio^2 = (n - 1) / n * (1 + mean(r)^2 / var_ddof0) >= (n - 1) / n.
    # The lower bound is attained when the window mean is zero; there is no
    # upper bound, since a large drift inflates the zero-mean estimator.
    ratio = (chosen / demeaned).dropna()
    lower_bound = np.sqrt((HORIZON - 1) / HORIZON)
    assert (ratio >= lower_bound - 1e-12).all()
    assert ratio.min() == pytest.approx(lower_bound, rel=0.05)
