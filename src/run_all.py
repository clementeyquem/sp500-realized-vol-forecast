"""Run the whole pipeline: evaluation, in-sample HAC fits, figures."""

from . import evaluate, figures, har_ols

if __name__ == "__main__":
    evaluate.main()
    har_ols.main()
    figures.main()
