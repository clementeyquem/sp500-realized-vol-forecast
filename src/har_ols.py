"""In-sample HAR-RV and HAR-RV + VIX fits with Newey-West standard errors.

Descriptive only (not part of the out-of-sample evaluation). The 21-day
targets of consecutive days overlap, so the OLS residuals are autocorrelated
and the usual standard errors are too small. Newey-West (HAC) with HORIZON
lags corrects for this.
"""

import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from .evaluate import TARGET_COL, build_dataset
from .features import FEATURES_HAR, FEATURES_HAR_VIX
from .target import HORIZON


def har_hac_fit(df: pd.DataFrame, features: list, last_year: int = 2007):
    """OLS of the forward target on `features`, 2000 to `last_year`, HAC errors."""
    sample = df[df.index.year <= last_year]
    X = add_constant(sample[features])
    return OLS(sample[TARGET_COL], X).fit(cov_type="HAC", cov_kwds={"maxlags": HORIZON})


def main() -> None:
    data = build_dataset()
    for name, feats in [("HAR-RV", FEATURES_HAR), ("HAR-RV + VIX", FEATURES_HAR_VIX)]:
        res = har_hac_fit(data, feats)
        print(f"\n{name}  (n={int(res.nobs)}, R2={res.rsquared:.3f}, HAC lags={HORIZON})")
        print(
            pd.DataFrame(
                {"coef": res.params, "se_HAC": res.bse, "t": res.tvalues, "p": res.pvalues}
            ).round(4).to_string()
        )


if __name__ == "__main__":
    main()
