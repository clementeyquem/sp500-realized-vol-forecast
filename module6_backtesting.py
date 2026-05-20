# =============================================================================
# MODULE 6 — Backtesting : De la Prédiction à la Stratégie de Trading
# =============================================================================
# pip install scikit-learn matplotlib pandas numpy yfinance
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DONNÉES (identique Module 5 — reprend le pipeline complet)
# =============================================================================
import yfinance as yf
spx = yf.download("^GSPC", start="2000-01-01", end="2024-12-31", auto_adjust=True)
vix = yf.download("^VIX",  start="2000-01-01", end="2024-12-31", auto_adjust=True)
spx_close = spx['Close'].squeeze()
vix_close = vix['Close'].squeeze()

log_ret      = np.log(spx_close / spx_close.shift(1))
realized_vol = log_ret.rolling(21).std() * np.sqrt(252) * 100

df = pd.DataFrame({
    'VIX':        vix_close,
    'RealizedVol':realized_vol,
    'SPX_ret':    log_ret,
}, index=vix_close.index).dropna()

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
for k in range(1, 22):
    df[f'VIX_lag{k}']  = df['VIX'].shift(k)
    df[f'DVIX_lag{k}'] = df['VIX'].diff(1).shift(k)

df['VIX_sq']      = df['VIX_lag1'] ** 2
df['VIX_sqrt']    = np.sqrt(df['VIX_lag1'].clip(0))
df['VIX_log']     = np.log(df['VIX_lag1'].clip(1))
df['VIX_high']    = (df['VIX_lag1'] > 30).astype(float)
df['VIX_extreme'] = (df['VIX_lag1'] > 50).astype(float)
df['VIX_range5']  = df['VIX_lag1'] - df['VIX_lag5']
df['VIX_range21'] = df['VIX_lag1'] - df['VIX_lag21']
df = df.dropna()

features_lin  = ['VIX_lag1', 'VIX_lag5', 'VIX_lag21', 'DVIX_lag1']
features_tree = ([f'VIX_lag{k}'  for k in range(1, 22)] +
                 [f'DVIX_lag{k}' for k in range(1, 22)] +
                 ['VIX_sq','VIX_sqrt','VIX_log',
                  'VIX_high','VIX_extreme','VIX_range5','VIX_range21'])

# =============================================================================
# 3. TRAIN / TEST SPLIT
# =============================================================================
X_lin  = df[features_lin].values
X_tree = df[features_tree].values
y      = df['RealizedVol'].values

split_idx  = int(len(df) * 0.80)
split_date = df.index[split_idx]

Xtr_lin,  Xte_lin  = X_lin[:split_idx],  X_lin[split_idx:]
Xtr_tree, Xte_tree = X_tree[:split_idx], X_tree[split_idx:]
ytr, yte = y[:split_idx], y[split_idx:]
dates_test = df.index[split_idx:]
vix_test   = df['VIX'].values[split_idx:]
spx_test   = df['SPX_ret'].values[split_idx:]

print(f"Train : {len(ytr)} obs → {split_date.date()}")
print(f"Test  : {len(yte)} obs → {df.index[-1].date()}")

# =============================================================================
# 4. ENTRAÎNEMENT DES MODÈLES
# =============================================================================
print("\nEntraînement des modèles...")

sc  = StandardScaler().fit(Xtr_lin)
ols = LinearRegression().fit(sc.transform(Xtr_lin), ytr)

rf  = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=20,
                             random_state=42, n_jobs=-1).fit(Xtr_tree, ytr)

gb  = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                                 min_samples_leaf=20, subsample=0.8,
                                 random_state=42).fit(Xtr_tree, ytr)

vol_pred_ols = ols.predict(sc.transform(Xte_lin))
vol_pred_rf  = rf.predict(Xte_tree)
vol_pred_gb  = gb.predict(Xte_tree)

# =============================================================================
# 5. FONCTION DE BACKTESTING
# =============================================================================
def backtest(vol_predicted, vol_realized, vix_current,
             model_name, threshold=0.0, transaction_cost=0.001):
    """
    Stratégie Long/Short Volatilité :
    - Si vol prédite > VIX + seuil → LONG VOL (vol sous-évaluée)
    - Si vol prédite < VIX - seuil → SHORT VOL (vol surévaluée)
    - Sinon → FLAT (signal trop faible)

    P&L = position × (vol_réalisée - VIX) / VIX × scaling
    """
    n = len(vol_predicted)
    pnl_daily = np.zeros(n)
    positions = np.zeros(n)

    for i in range(n):
        # ── 1. Signal ─────────────────────────────
        signal = vol_predicted[i] - vix_current[i]

        if   signal >  threshold:  pos =  1.0   # LONG VOL
        elif signal < -threshold:  pos = -1.0   # SHORT VOL
        else:                      pos =  0.0   # FLAT

        positions[i] = pos

        # ── 2. Coût de transaction ─────────────────
        if i > 0 and pos != positions[i-1]:
            pnl_daily[i] -= transaction_cost

        # ── 3. P&L du jour ────────────────────────
        # On gagne si la direction de la surprise de vol est correcte
        if pos != 0:
            vol_surprise    = (vol_realized[i] - vix_current[i]) / (vix_current[i] + 1e-6)
            pnl_daily[i]   += pos * vol_surprise * 0.5

    # ── Métriques ─────────────────────────────────
    cum_pnl   = np.cumsum(pnl_daily)
    active    = pnl_daily[positions != 0]
    daily_std = np.std(pnl_daily) if np.std(pnl_daily) > 0 else 1e-6
    sharpe    = (np.mean(pnl_daily) / daily_std) * np.sqrt(252)
    drawdown  = cum_pnl - np.maximum.accumulate(cum_pnl)
    max_dd    = np.min(drawdown)
    win_rate  = np.mean(active > 0) if len(active) > 0 else 0

    print(f"\n{'─'*40}")
    print(f"  {model_name}")
    print(f"  Rendement total : {cum_pnl[-1]*100:+.2f}%")
    print(f"  Sharpe ratio    : {sharpe:.3f}")
    print(f"  Max drawdown    : {max_dd*100:.2f}%")
    print(f"  Win rate        : {win_rate*100:.1f}%")
    print(f"  Nb trades       : {int(np.sum(np.diff(positions) != 0))}")
    print(f"  Long  : {int(np.sum(positions==1))} jours  |  Short : {int(np.sum(positions==-1))} jours")

    return {
        'pnl':      pnl_daily,
        'cum_pnl':  cum_pnl,
        'drawdown': drawdown,
        'pos':      positions,
        'sharpe':   sharpe,
        'maxdd':    max_dd,
        'winrate':  win_rate,
        'total':    cum_pnl[-1],
        'name':     model_name,
    }

# =============================================================================
# 6. LANCEMENT DES BACKTESTS
# =============================================================================
print("\n=== BACKTESTING — Stratégie Long/Short Vol ===")
bt_ols = backtest(vol_pred_ols, yte, vix_test, "OLS")
bt_rf  = backtest(vol_pred_rf,  yte, vix_test, "Random Forest")
bt_gb  = backtest(vol_pred_gb,  yte, vix_test, "Gradient Boosting")

# Benchmark Buy & Hold SPX
bh_cum    = np.cumsum(spx_test)
bh_sharpe = (np.mean(spx_test) / np.std(spx_test)) * np.sqrt(252)
print(f"\n{'─'*40}")
print(f"  Buy & Hold SPX")
print(f"  Rendement total : {bh_cum[-1]*100:+.2f}%")
print(f"  Sharpe ratio    : {bh_sharpe:.3f}")

# =============================================================================
# 7. MÉTRIQUES FINANCIÈRES — RAPPEL
# =============================================================================
print("\n" + "="*60)
print("MÉTRIQUES CLÉS")
print("="*60)
print("""
Sharpe Ratio = E[r] / σ(r) × √252
  → Rendement moyen / écart-type × annualisation
  → > 1 = stratégie viable, > 2 = excellent

Max Drawdown = pire perte depuis un pic
  → Mesure le risque ressenti par l'investisseur
  → max_dd = min(cum_pnl - max_cumulatif)

Win Rate = % de jours/trades gagnants
  → Insuffisant seul (peut être < 50% et rentable si gains > pertes)
  → Toujours combiner avec le ratio gain moyen / perte moyenne
""")

# =============================================================================
# 8. VISUALISATIONS
# =============================================================================
PLT_BG = '#F8FAFC'
fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor=PLT_BG)
fig.suptitle('Module 6 — Backtesting : Stratégie Long/Short Volatilité',
             fontsize=13, fontweight='bold', color='#0D1117', y=1.01)

# ── Fig 1 : Signal de trading ─────────────────
ax = axes[0, 0]
ax.plot(dates_test, vix_test,     color='#6B7280', lw=1.2, alpha=0.7, label='VIX (vol implicite)')
ax.plot(dates_test, vol_pred_gb,  color='#7C3AED', lw=1.5, label='GBM (vol prédite)')
ax.fill_between(dates_test, vix_test, vol_pred_gb,
                where=vol_pred_gb > vix_test, alpha=0.25, color='#10B981',
                label='LONG VOL')
ax.fill_between(dates_test, vix_test, vol_pred_gb,
                where=vol_pred_gb < vix_test, alpha=0.25, color='#EF4444',
                label='SHORT VOL')
ax.set_title('Signal GBM : Vol prédite vs VIX', fontsize=10, fontweight='bold')
ax.set_ylabel('Volatilité (%)', fontsize=9)
ax.legend(fontsize=8, ncol=2); ax.set_facecolor(PLT_BG)
ax.grid(True, alpha=0.3, linestyle='--'); ax.spines[['top','right']].set_visible(False)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ── Fig 2 : P&L cumulatif ─────────────────────
ax = axes[0, 1]
ax.plot(dates_test, bh_cum*100,           color='#6B7280', lw=1.5, linestyle='--',
        alpha=0.7, label=f'B&H SPX  Sharpe={bh_sharpe:.2f}')
ax.plot(dates_test, bt_ols['cum_pnl']*100, color='#2563EB', lw=1.5,
        label=f'OLS  Sharpe={bt_ols["sharpe"]:.2f}')
ax.plot(dates_test, bt_rf['cum_pnl']*100,  color='#065F46', lw=1.8,
        label=f'RF   Sharpe={bt_rf["sharpe"]:.2f}')
ax.plot(dates_test, bt_gb['cum_pnl']*100,  color='#7C3AED', lw=2,
        label=f'GBM  Sharpe={bt_gb["sharpe"]:.2f}')
ax.axhline(0, color='#374151', lw=0.8, linestyle='--')
ax.set_title('P&L Cumulatif (%)', fontsize=10, fontweight='bold')
ax.set_ylabel('Rendement cumulatif (%)', fontsize=9)
ax.legend(fontsize=8); ax.set_facecolor(PLT_BG)
ax.grid(True, alpha=0.3, linestyle='--'); ax.spines[['top','right']].set_visible(False)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ── Fig 3 : Drawdowns ─────────────────────────
ax = axes[1, 0]
for bt, col, lbl in [(bt_ols,'#2563EB','OLS'), (bt_rf,'#065F46','RF'), (bt_gb,'#7C3AED','GBM')]:
    ax.fill_between(dates_test, bt['drawdown']*100, 0, alpha=0.3, color=col)
    ax.plot(dates_test, bt['drawdown']*100, color=col, lw=1.2,
            label=f'{lbl}  MaxDD={bt["maxdd"]*100:.1f}%')
ax.set_title('Drawdowns (%)', fontsize=10, fontweight='bold')
ax.set_ylabel('Drawdown (%)', fontsize=9)
ax.legend(fontsize=8); ax.set_facecolor(PLT_BG)
ax.grid(True, alpha=0.3, linestyle='--'); ax.spines[['top','right']].set_visible(False)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ── Fig 4 : Comparaison Sharpe ────────────────
ax = axes[1, 1]
models   = ['OLS', 'RF', 'GBM', 'B&H']
sharpes  = [bt_ols['sharpe'], bt_rf['sharpe'], bt_gb['sharpe'], bh_sharpe]
bar_cols = ['#2563EB', '#065F46', '#7C3AED', '#6B7280']
bars = ax.bar(models, sharpes, color=bar_cols, alpha=0.80, edgecolor='white', width=0.6)
for bar, val in zip(bars, sharpes):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height() + (0.02 if val >= 0 else -0.08),
            f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(0,   color='#374151', lw=0.8, linestyle='--')
ax.axhline(1.0, color='#10B981', lw=1.2, linestyle=':', alpha=0.7, label='Seuil viable (Sharpe=1)')
ax.axhline(2.0, color='#F59E0B', lw=1.2, linestyle=':', alpha=0.7, label='Seuil excellent (Sharpe=2)')
ax.set_title('Sharpe Ratio par modèle', fontsize=10, fontweight='bold')
ax.set_ylabel('Sharpe (annualisé)', fontsize=9)
ax.legend(fontsize=8); ax.set_facecolor(PLT_BG)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('module6_backtest.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGraphique sauvegardé : module6_backtest.png")

# =============================================================================
# 9. RÉCAPITULATIF FINAL DU COURS
# =============================================================================
print("\n" + "="*60)
print("RÉCAPITULATIF DU COURS COMPLET")
print("="*60)
print("""
Module 1 → Import données, vol réalisée, R²
Module 2 → Lags, régression multiple, look-ahead bias
Module 3 → TimeSeriesSplit, walk-forward validation
Module 4 → Ridge & Lasso, régularisation, sélection features
Module 5 → Random Forest, Gradient Boosting, non-linéarité
Module 6 → Backtesting, Sharpe, drawdown, stratégie vol
""")
