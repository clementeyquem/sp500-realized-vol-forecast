"""Walk-forward evaluation, out-of-sample metrics and the results table.

Protocol (CLAUDE.md section 6.2):
  - expanding training window, retrained once a year
  - test on the following calendar year, from 2008 to 2024
  - a gap of HORIZON trading days between the end of train and the start of
    test, because consecutive 21-day targets overlap

Metrics: RMSE, MAE, out-of-sample R2 relative to the persistence baseline,
and QLIKE (Patton, 2011), which is computed on variance rather than
volatility.
"""

import numpy as np
import pandas as pd
# Narrow import on purpose: statsmodels.api pulls in the graphics subpackage,
# which is not needed here.
from scipy import stats
from statsmodels.regression.linear_model import OLS

from .data import load_close_series, log_returns
from .features import build_features
from .models import ALPHA_GRID, BASELINE_FOR_R2_OOS, build_models
from .target import HORIZON, forward_realized_vol

FIRST_TEST_YEAR = 2008
LAST_TEST_YEAR = 2024
TARGET_COL = "target_h21"

# Forecasts are floored right after prediction (so RMSE, MAE and QLIKE all use
# the same numbers), because QLIKE is undefined for a non-positive forecast. The number of clipped values is
# reported: it is a diagnostic, not something to hide.
MIN_VOL_FORECAST = 1.0  # volatility points


def build_dataset() -> pd.DataFrame:
    """Features and forward target, aligned, with incomplete rows dropped."""
    px = load_close_series()
    r = log_returns(px["spx"])
    X = build_features(r, px["vix"])
    y = forward_realized_vol(r).rename(TARGET_COL)
    return pd.concat([X, y], axis=1).dropna()


def walk_forward_predictions(df: pd.DataFrame):
    """Out-of-sample predictions for every model, one column per model.

    For each test year the model is refitted on all observations up to the
    start of that year minus HORIZON trading days.

    Returns (preds, alphas, n_clipped):
      - preds: forecasts floored at MIN_VOL_FORECAST
      - alphas: penalty retained by the inner CV, one row per fold and model
      - n_clipped: number of forecasts that were raised to the floor
    """
    models = build_models()
    preds = pd.DataFrame(index=df.index, columns=list(models), dtype=float)
    n_clipped = dict.fromkeys(models, 0)
    alpha_rows = []
    y = df[TARGET_COL].to_numpy()

    for year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        test_mask = df.index.year == year
        if not test_mask.any():
            continue
        test_pos = np.flatnonzero(test_mask)
        # Purge: drop the HORIZON observations immediately before the test set,
        # whose target windows overlap the test period.
        train_end = test_pos[0] - HORIZON
        if train_end <= 0:
            continue
        train_pos = np.arange(train_end)

        for name, (estimator, feature_names) in models.items():
            X_train = df.iloc[train_pos][feature_names].to_numpy()
            X_test = df.iloc[test_pos][feature_names].to_numpy()
            estimator.fit(X_train, y[train_pos])
            raw = estimator.predict(X_test)
            n_clipped[name] += int(np.sum(raw < MIN_VOL_FORECAST))
            preds.iloc[test_pos, preds.columns.get_loc(name)] = np.clip(
                raw, MIN_VOL_FORECAST, None
            )
            if name in ("Ridge", "Lasso"):
                alpha = float(estimator[-1].alpha_)
                alpha_rows.append(
                    {
                        "year": year,
                        "model": name,
                        "alpha": alpha,
                        "on_lower_bound": bool(np.isclose(alpha, ALPHA_GRID[0])),
                    }
                )

    return preds, pd.DataFrame(alpha_rows), pd.Series(n_clipped, name="n_clipped")


def qlike(y_true_vol: np.ndarray, y_pred_vol: np.ndarray) -> float:
    """QLIKE loss on variance (Patton, 2011), in its non-negative form.

        QLIKE = mean( s2/h - log(s2/h) - 1 )

    with s2 the realized variance proxy and h the forecast variance. The loss
    is zero for a perfect forecast and robust to the fact that the true
    variance is latent. Inputs are annualized volatility points.
    """
    s2 = (y_true_vol / 100.0) ** 2
    h = (y_pred_vol / 100.0) ** 2  # forecasts are already floored
    ratio = s2 / h
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _metrics(y_true, y_pred, sse_baseline) -> dict:
    err = y_true - y_pred
    sse = float(np.sum(err**2))
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "R2_OOS_vs_persistence": 1.0 - sse / sse_baseline,
        "QLIKE": qlike(y_true, y_pred),
    }


def results_table(
    df: pd.DataFrame, preds: pd.DataFrame, n_clipped: pd.Series
) -> pd.DataFrame:
    """One row per model, metrics aggregated over the whole test period."""
    test = preds.dropna(how="all").index
    y_true = df.loc[test, TARGET_COL].to_numpy()

    baseline = preds.loc[test, BASELINE_FOR_R2_OOS].to_numpy()
    sse_baseline = float(np.sum((y_true - baseline) ** 2))

    rows = {
        name: _metrics(y_true, preds.loc[test, name].to_numpy(), sse_baseline)
        for name in preds.columns
    }
    out = pd.DataFrame(rows).T
    out["n_clipped"] = n_clipped.astype(int)
    out.index.name = "model"
    return out


def regime_label(index: pd.DatetimeIndex) -> pd.Series:
    """Split the test period into the two stress episodes and the rest."""
    years = index.year
    labels = np.where(
        np.isin(years, [2008, 2009]),
        "2008-2009 (GFC)",
        np.where(years == 2020, "2020 (Covid)", "Other years"),
    )
    return pd.Series(labels, index=index, name="regime")


def results_by_regime(df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """RMSE and QLIKE per model and per regime."""
    test = preds.dropna(how="all").index
    regimes = regime_label(test)
    rows = []

    for regime, idx in regimes.groupby(regimes).groups.items():
        y_true = df.loc[idx, TARGET_COL].to_numpy()
        baseline = preds.loc[idx, BASELINE_FOR_R2_OOS].to_numpy()
        sse_baseline = float(np.sum((y_true - baseline) ** 2))
        for name in preds.columns:
            m = _metrics(y_true, preds.loc[idx, name].to_numpy(), sse_baseline)
            rows.append(
                {
                    "regime": regime,
                    "model": name,
                    "n": len(idx),
                    "RMSE": m["RMSE"],
                    "QLIKE": m["QLIKE"],
                    "R2_OOS_vs_persistence": m["R2_OOS_vs_persistence"],
                }
            )
    return pd.DataFrame(rows)


def qlike_loss(y_true_vol: np.ndarray, y_pred_vol: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE, used for the Diebold-Mariano loss differential."""
    ratio = (y_true_vol / y_pred_vol) ** 2
    return ratio - np.log(ratio) - 1.0


def diebold_mariano(y_true, pred_a, pred_b, loss: str, lags: int = HORIZON - 1) -> dict:
    """Diebold-Mariano test of model A against model B.

    The loss differential d = L(A) - L(B) is autocorrelated because the 21-day
    targets overlap, so the variance of its mean is Newey-West (Bartlett
    kernel) with HORIZON - 1 = 20 lags. The statistic gets the
    Harvey-Leybourne-Newbold small-sample correction and is compared with a
    Student t(n - 1). A negative statistic means A has the lower loss.
    """
    if loss == "squared":
        d = (y_true - pred_a) ** 2 - (y_true - pred_b) ** 2
    elif loss == "qlike":
        d = qlike_loss(y_true, pred_a) - qlike_loss(y_true, pred_b)
    else:
        raise ValueError(loss)
    n, h = len(d), HORIZON
    fit = OLS(d, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    dm = float(fit.tvalues[0])
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln
    return {
        "loss": loss,
        "mean_loss_diff": float(np.mean(d)),
        "dm_stat_hln": dm_hln,
        "p_value": float(2 * stats.t.sf(abs(dm_hln), df=n - 1)),
        "nw_lags": lags,
        "n": n,
    }


# The three comparisons are fixed in advance. "Best ML model" is the one with
# the lowest aggregate RMSE among the four ML models (rule fixed before the
# tests are read; it is a selection on the test sample, so it is reported).
ML_MODELS = ["Ridge", "Lasso", "Random Forest", "Gradient Boosting"]


def dm_tests(df: pd.DataFrame, preds: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """The three planned Diebold-Mariano comparisons, on squared error and QLIKE."""
    best_ml = table.loc[ML_MODELS, "RMSE"].idxmin()
    pairs = [
        ("HAR-RV + VIX", "HAR-RV"),
        (best_ml, "HAR-RV + VIX"),
        ("HAR-RV", "Persistence"),
    ]
    test = preds.dropna(how="all").index
    y_true = df.loc[test, TARGET_COL].to_numpy()
    rows = []
    for a, b in pairs:
        for loss in ("squared", "qlike"):
            res = diebold_mariano(
                y_true, preds.loc[test, a].to_numpy(), preds.loc[test, b].to_numpy(), loss
            )
            rows.append({"model_a": a, "model_b": b, **res})
    return pd.DataFrame(rows)


def main() -> None:
    """Run the walk-forward evaluation and write every file under results/."""
    from pathlib import Path

    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)

    df = build_dataset()
    preds, alphas, n_clipped = walk_forward_predictions(df)
    table = results_table(df, preds, n_clipped)

    table.to_csv(out / "results.csv")
    results_by_regime(df, preds).to_csv(out / "results_by_regime.csv", index=False)
    # First line is a comment: read with pd.read_csv(..., comment="#").
    with open(out / "diebold_mariano.csv", "w", newline="") as f:
        f.write("# sign convention: dm_stat_hln < 0 means model_a has a lower loss than model_b\n")
        dm_tests(df, preds, table).to_csv(f, index=False)
    alphas.to_csv(out / "alphas_by_fold.csv", index=False)

    test = preds.dropna(how="all").index
    pred_out = preds.loc[test].copy()
    pred_out.insert(0, "realized", df.loc[test, TARGET_COL])
    pred_out.index.name = "date"
    pred_out.to_csv(out / "predictions.csv")

    print(table.round(4).to_string())
    print("\nAlpha on lower bound:", int(alphas["on_lower_bound"].sum()), "of", len(alphas))


if __name__ == "__main__":
    main()
