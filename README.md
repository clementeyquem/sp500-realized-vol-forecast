# VIX → S&P 500 Volatility Prediction — Linear Regression Project

> Predicting the **realized volatility of the S&P 500** from the **VIX index** using progressively advanced regression models — from simple OLS to Random Forest and Gradient Boosting.
>
> ---
>
> ## Project Overview
>
> This project is structured as a **7-module learning path** exploring how the VIX (implied volatility index) can be used to predict the future realized volatility of the S&P 500 (GSPX). Each module adds complexity and rigor to the modeling approach.
>
> | Module | Notebook | Script | Description |
> |--------|----------|--------|-------------|
> | 1 | `Linear_Regression_Module_1.ipynb` | — | OLS baseline: VIX(t) → RealizedVol(t+1), assumptions check |
> | 2 | `Linear_Regression_Module_2.ipynb` | `module2_vix_regression.py` | Multiple regression with lags (VIX t-1 to t-4), no look-ahead bias |
> | 3 | `Linear_Regression_Module_3.ipynb` | `module3_timeseries_cv.py` | Temporal train/test split, time-series cross-validation |
> | 4 | `Linear_Regression_Module_4_Ridge_Lasso.ipynb` | `module4_ridge_lasso.py` | Regularization: Ridge & Lasso, hyperparameter tuning |
> | 5 | `Linear_Regression_Module_5_Random_Forest_Gradient_Boosting.ipynb` | `module5_random_forest_boosting.py` | Non-linear models: Random Forest & XGBoost |
> | 6 | — | `module6_backtesting.py` | Simple backtesting of volatility signals |
> | 7 | — | `module7_institutional_backtest.py` | Institutional-grade backtest with transaction costs |
>
> ---
>
> ## Key Results
>
> - **Module 1** — OLS R² ≈ 0.55 on test set; heteroskedasticity and autocorrelation identified
> - - **Module 2** — Multi-lag model improves R² to ~0.62; VIX(t-1) is the strongest predictor
>   - - **Module 3** — Temporal CV confirms model stability; avoids data leakage
>     - - **Module 4** — Lasso selects 3 key lags; Ridge reduces variance; best test R² ≈ 0.64
>       - - **Module 5** — Random Forest achieves R² ≈ 0.71; Gradient Boosting reaches ~0.74
>        
>         - ---
>
> ## Repository Structure
>
> ```
> Linear_Regression/
> ├── notebooks/
> │   ├── Linear_Regression_Module_1.ipynb          # OLS baseline
> │   ├── Linear_Regression_Module_2.ipynb          # Multi-lag regression
> │   ├── Linear_Regression_Module_3.ipynb          # Temporal CV
> │   ├── Linear_Regression_Module_4_Ridge_Lasso.ipynb
> │   └── Linear_Regression_Module_5_Random_Forest_Gradient_Boosting.ipynb
> ├── scripts/
> │   ├── module2_vix_regression.py
> │   ├── module3_timeseries_cv.py
> │   ├── module4_ridge_lasso.py
> │   ├── module5_random_forest_boosting.py
> │   ├── module6_backtesting.py
> │   └── module7_institutional_backtest.py
> ├── results/
> │   ├── module4_results.png
> │   └── module5_results.png
> └── README.md
> ```
>
> > **Note:** Course PDFs (`cours_vix_*.pdf`) are reference materials from the course curriculum.
> >
> > ---
> >
> > ## Tech Stack
> >
> > - **Python 3.10+**
> > - - `pandas`, `numpy` — data manipulation
> >   - - `scikit-learn` — regression models, Ridge, Lasso, cross-validation
> >     - - `xgboost` / `sklearn.ensemble` — Random Forest, Gradient Boosting
> >       - - `matplotlib`, `seaborn` — visualization
> >         - - `yfinance` — financial data download (optional, synthetic data as fallback)
> >          
> >           - ---
> >
> > ## Getting Started
> >
> > ```bash
> > # Clone the repository
> > git clone https://github.com/clementeyquem/Linear_Regression.git
> > cd Linear_Regression
> >
> > # Install dependencies
> > pip install pandas numpy matplotlib scikit-learn xgboost yfinance seaborn
> >
> > # Run a module script
> > python module2_vix_regression.py
> >
> > # Or open notebooks in Jupyter
> > jupyter notebook
> > ```
> >
> > ---
> >
> > ## Methodology Notes
> >
> > - **No look-ahead bias**: all features use lagged values only (VIX at t-1, t-2, ...)
> > - - **Temporal train/test split**: 80% training (2018–2022), 20% test (2023–2024)
> >   - - **Target variable**: 21-day rolling realized volatility of S&P 500 log returns, annualized
> >     - - **Data source**: Yahoo Finance (`^GSPC`, `^VIX`) or synthetic Ornstein-Uhlenbeck process
> >      
> >       - ---
> >
> > *Project by [Clément Eyquem](https://github.com/clementeyquem)*
