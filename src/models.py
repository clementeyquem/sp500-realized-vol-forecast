"""Baselines and models, ordered from simplest to most complex.

The two naive baselines (persistence, raw VIX) fit nothing: they copy a
column. They are the hardest to beat and must be reported alongside every
other model (see CLAUDE.md section 4).

Regularized and tree models see the full feature set. Alpha and all
hyperparameters are chosen inside the training window only, using a
TimeSeriesSplit with a gap of HORIZON days so that overlapping targets cannot
leak across the inner folds.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    FEATURES_ALL,
    FEATURES_HAR,
    FEATURES_HAR_VIX,
    FEATURES_PERSISTENCE,
    FEATURES_VIX,
)
from .target import HORIZON

SEED = 42
INNER_SPLITS = 5
# One log grid for both penalties: 1e-4 to 1e3. If the retained alpha sits on
# the lower bound, regularization is doing nothing (the problem of the v1).
ALPHA_GRID = np.logspace(-4, 3, 100)


def inner_cv() -> TimeSeriesSplit:
    """Time-series CV for hyperparameter selection inside the training window.

    gap=HORIZON purges the overlap between consecutive 21-day targets, so the
    last training observations of a fold do not share their target window with
    the first validation observations.
    """
    return TimeSeriesSplit(n_splits=INNER_SPLITS, gap=HORIZON)


class CopyColumn(BaseEstimator, RegressorMixin):
    """Predict a single input column unchanged. Fits nothing.

    Used for the persistence baseline (copy past 21-day realized vol) and the
    raw VIX baseline (copy the VIX level).
    """

    def fit(self, X, y=None):
        if X.shape[1] != 1:
            raise ValueError(f"CopyColumn expects one column, got {X.shape[1]}")
        return self

    def predict(self, X):
        return np.asarray(X).ravel()


def build_models() -> dict:
    """Return {name: (estimator, feature_names)}, in reporting order."""
    return {
        # --- Baselines -------------------------------------------------
        "Persistence": (CopyColumn(), FEATURES_PERSISTENCE),
        "VIX (raw)": (CopyColumn(), FEATURES_VIX),
        "VIX (calibrated)": (LinearRegression(), FEATURES_VIX),
        "HAR-RV": (LinearRegression(), FEATURES_HAR),
        "HAR-RV + VIX": (LinearRegression(), FEATURES_HAR_VIX),
        # --- Regularized linear models ---------------------------------
        "Ridge": (
            make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=ALPHA_GRID, cv=inner_cv()),
            ),
            FEATURES_ALL,
        ),
        "Lasso": (
            make_pipeline(
                StandardScaler(),
                LassoCV(
                    alphas=ALPHA_GRID,
                    cv=inner_cv(),
                    # The VIX transforms are strongly collinear, so coordinate
                    # descent needs a generous iteration budget at small alpha.
                    max_iter=500_000,
                    tol=1e-3,
                    random_state=SEED,
                ),
            ),
            FEATURES_ALL,
        ),
        # --- Tree ensembles --------------------------------------------
        # No scaling: trees are invariant to monotone rescaling of inputs.
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=20,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=-1,
            ),
            FEATURES_ALL,
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=SEED,
            ),
            FEATURES_ALL,
        ),
    }


BASELINE_FOR_R2_OOS = "Persistence"
REFERENCE_MODEL = "HAR-RV"
