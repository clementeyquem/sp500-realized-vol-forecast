"""Figures written to results/, from the CSV files produced by evaluate.py.

Plain black-and-white style: models are told apart by line style or hatching,
not by colour alone.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import regime_label

RESULTS = Path(__file__).resolve().parent.parent / "results"
DPI = 150
BEST_ML = "Ridge"  # lowest aggregate RMSE among the ML models (see evaluate.py)
HATCHES = ["", "//", "..", "xx", "\\\\", "++", "--", "oo", "**"]


def _style():
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": DPI}
    )


def fig_forecasts(preds: pd.DataFrame) -> None:
    """Realized vs HAR vs best ML model, with a zoom on 2020."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    for ax, sub, title in [
        (axes[0], preds, "Realized volatility and 21-day-ahead forecasts, 2008-2024"),
        (axes[1], preds[preds.index.year == 2020], "Zoom on 2020"),
    ]:
        ax.plot(sub.index, sub["realized"], color="black", lw=1.3, label="Realized (next 21 days)")
        ax.plot(sub.index, sub["HAR-RV"], color="0.35", lw=1, ls="--", label="HAR-RV")
        ax.plot(sub.index, sub[BEST_ML], color="0.55", lw=1, ls=":", label=BEST_ML)
        ax.set_title(title)
        ax.set_ylabel("Annualized volatility (points)")
        ax.legend(frameon=False)
    axes[1].set_xlabel("Forecast date")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_forecasts.png", dpi=DPI)
    plt.close(fig)


def fig_metrics(table: pd.DataFrame) -> None:
    """RMSE and QLIKE bars, one per model."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, col in zip(axes, ["RMSE", "QLIKE"]):
        ax.bar(range(len(table)), table[col], color="white", edgecolor="black",
               hatch=None)
        for i, bar in enumerate(ax.patches):
            bar.set_hatch(HATCHES[i % len(HATCHES)])
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels(table.index, rotation=45, ha="right")
        ax.set_title(f"{col}, out-of-sample 2008-2024"
                     + (" (vol points)" if col == "RMSE" else " (on variance)"))
        ax.set_ylabel(col)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_rmse_qlike.png", dpi=DPI)
    plt.close(fig)


def fig_yearly(preds: pd.DataFrame) -> None:
    """Annual RMSE for HAR, HAR+VIX and the best ML model."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for model, ls, mk in [("HAR-RV", "--", "o"), ("HAR-RV + VIX", "-", "s"), (BEST_ML, ":", "^")]:
        err = (preds["realized"] - preds[model]) ** 2
        rmse = np.sqrt(err.groupby(preds.index.year).mean())
        ax.plot(rmse.index, rmse.values, color="black", ls=ls, marker=mk, ms=4, lw=1.2, label=model)
    ax.set_title("RMSE by test year")
    ax.set_xlabel("Test year")
    ax.set_ylabel("RMSE (volatility points)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_rmse_by_year.png", dpi=DPI)
    plt.close(fig)


def fig_regimes(preds: pd.DataFrame) -> None:
    """RMSE by regime for six representative models."""
    models = ["Persistence", "VIX (calibrated)", "HAR-RV", "HAR-RV + VIX", "Ridge", "Random Forest"]
    regimes = ["2008-2009 (GFC)", "2020 (Covid)", "Other years"]
    reg = regime_label(preds.index)
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, m in enumerate(models):
        vals = [np.sqrt(((preds["realized"] - preds[m])[reg == r] ** 2).mean()) for r in regimes]
        ax.bar(np.arange(3) + j * width, vals, width, color="white", edgecolor="black",
               hatch=HATCHES[j], label=m)
    ax.set_xticks(np.arange(3) + 0.4 - width / 2)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("RMSE (volatility points)")
    ax.set_title("RMSE by regime")
    ax.set_ylim(0, 31)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig4_rmse_by_regime.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    _style()
    preds = pd.read_csv(RESULTS / "predictions.csv", index_col=0, parse_dates=True)
    table = pd.read_csv(RESULTS / "results.csv", index_col=0)
    fig_forecasts(preds)
    fig_metrics(table)
    fig_yearly(preds)
    fig_regimes(preds)


if __name__ == "__main__":
    main()
