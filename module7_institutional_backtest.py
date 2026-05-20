# =============================================================================
# MODULE 7 — Backtesting Institutionnel
# Black-Scholes · Prix d'options réels · Walk-Forward · Gestion du risque
# =============================================================================
# pip install scikit-learn matplotlib pandas numpy yfinance scipy
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# BLOC 1 — BLACK-SCHOLES PRICING ENGINE
# =============================================================================

def bs_price(S, K, T, r, sigma, option_type='call'):
    """Prix Black-Scholes d'un call ou put."""
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def straddle_price(S, K, T, r, sigma):
    """Prix straddle ATM = call + put."""
    return bs_price(S, K, T, r, sigma, 'call') + bs_price(S, K, T, r, sigma, 'put')

def bs_vega(S, K, T, r, sigma):
    """Vega : sensibilité au prix à sigma."""
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol(market_price, S, K, T, r, option_type='call'):
    """Vol implicite par inversion Black-Scholes."""
    try:
        f = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price
        return brentq(f, 1e-6, 5.0)
    except:
        return np.nan

# =============================================================================
# BLOC 2 — DONNÉES (yfinance 2000-2024)
# =============================================================================
import yfinance as yf
print("Téléchargement des données...")
spx = yf.download("^GSPC", start="2000-01-01", end="2024-12-31", auto_adjust=True, progress=False)
vix = yf.download("^VIX",  start="2000-01-01", end="2024-12-31", auto_adjust=True, progress=False)
spx_close = spx['Close'].squeeze()
vix_close  = vix['Close'].squeeze()

log_ret      = np.log(spx_close / spx_close.shift(1))
realized_vol = log_ret.rolling(21).std() * np.sqrt(252) * 100

# Taux sans risque (approximation : 1% avant 2022, 5% après)
dates_all = spx_close.index
rf_rate   = pd.Series(np.where(dates_all < pd.Timestamp("2022-03-01"), 0.01, 0.05),
                       index=dates_all)

# Bid-Ask spread implicite (modélisé)
vix_chg    = vix_close.diff().abs().fillna(0)
bid_ask_vol = (0.5 + 0.08*vix_close + 0.3*vix_chg).clip(0.3, 8.0)

df = pd.DataFrame({
    'VIX':        vix_close,
    'RealizedVol':realized_vol,
    'SPX':        spx_close,
    'SPX_ret':    log_ret,
    'RF':         rf_rate,
    'BidAskVol':  bid_ask_vol,
}).dropna()

# =============================================================================
# BLOC 3 — FEATURE ENGINEERING
# =============================================================================
for k in range(1, 22):
    df[f'VIX_lag{k}']  = df['VIX'].shift(k)
    df[f'DVIX_lag{k}'] = df['VIX'].diff(1).shift(k)

df['VIX_sq']      = df['VIX_lag1']**2
df['VIX_sqrt']    = np.sqrt(df['VIX_lag1'].clip(0))
df['VIX_log']     = np.log(df['VIX_lag1'].clip(1))
df['VIX_high']    = (df['VIX_lag1'] > 30).astype(float)
df['VIX_extreme'] = (df['VIX_lag1'] > 50).astype(float)
df['VIX_range5']  = df['VIX_lag1'] - df['VIX_lag5']
df['VIX_range21'] = df['VIX_lag1'] - df['VIX_lag21']
df['VIX_vol21']   = df['VIX'].rolling(21).std()
df['VIX_skew']    = df['VIX_lag1'] / (df['VIX_lag21'] + 1e-6)
df = df.dropna()

features = ([f'VIX_lag{k}'  for k in range(1,22)] +
            [f'DVIX_lag{k}' for k in range(1,22)] +
            ['VIX_sq','VIX_sqrt','VIX_log','VIX_high','VIX_extreme',
             'VIX_range5','VIX_range21','VIX_vol21','VIX_skew'])

X = df[features].values
y = df['RealizedVol'].values
print(f"Dataset : {len(df)} observations | {len(features)} features")

# =============================================================================
# BLOC 4 — WALK-FORWARD (re-entraînement annuel)
# =============================================================================
INITIAL_TRAIN = 1260   # 5 ans
RETRAIN_FREQ  = 252    # annuel

print(f"\nWalk-forward : {INITIAL_TRAIN}j initial, re-entraînement tous les {RETRAIN_FREQ}j")
vol_predictions = np.full(len(df), np.nan)
retrain_dates   = []

i = INITIAL_TRAIN
while i < len(df):
    gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.07,
                                    max_depth=3, min_samples_leaf=25,
                                    subsample=0.8, random_state=42)
    gb.fit(X[:i], y[:i])
    retrain_dates.append(df.index[i])
    end = min(i + RETRAIN_FREQ, len(df))
    for j in range(i, end):
        vol_predictions[j] = gb.predict(X[j:j+1])[0]
    i = end

df['VolPred'] = vol_predictions
df_bt = df[df['VolPred'].notna()].copy()
print(f"Backtest : {df_bt.index[0].date()} → {df_bt.index[-1].date()} | {len(df_bt)} obs")
print(f"Re-entraînements : {len(retrain_dates)}")

# =============================================================================
# BLOC 5 — BACKTESTING INSTITUTIONNEL
# =============================================================================
CAPITAL_INIT  = 10_000_000   # $10M
MAX_POSITION  = 0.02         # 2% du capital par trade
MIN_SIGNAL    = 2.5          # seuil signal (pts de vol)
STOP_LOSS     = 0.015        # -1.5% du capital max par trade
MAX_DD_LIMIT  = 0.15         # arrêt si drawdown > 15%
HOLD_PERIOD   = 21           # jours avant expiration

capital        = CAPITAL_INIT
capital_series = [capital]
trade_log      = []
n_bt           = len(df_bt)
dates_bt       = df_bt.index
trading_halted = False

print(f"\nCapital : ${CAPITAL_INIT:,.0f} | Signal min : {MIN_SIGNAL}pts | DD limit : {MAX_DD_LIMIT*100:.0f}%")

for i in range(n_bt - HOLD_PERIOD):
    if trading_halted:
        capital_series.append(capital)
        continue

    row     = df_bt.iloc[i]
    S_spot  = row['SPX']
    K       = S_spot          # straddle ATM
    vol_impl = row['VIX'] / 100.0
    vol_pred = row['VolPred'] / 100.0
    T        = HOLD_PERIOD / 252.0
    r        = row['RF']
    bid_ask  = row['BidAskVol'] / 100.0

    # ── Signal
    signal    = row['VolPred'] - row['VIX']
    if   signal >  MIN_SIGNAL:  direction =  1
    elif signal < -MIN_SIGNAL:  direction = -1
    else:
        capital_series.append(capital)
        continue

    # ── Prix avec bid-ask
    vol_entry = np.clip(vol_impl + direction * bid_ask/2, 0.05, 2.0)
    premium   = straddle_price(S_spot, K, T, r, vol_entry)
    if premium <= 0:
        capital_series.append(capital)
        continue

    # ── Taille et coûts
    notional    = capital * MAX_POSITION
    n_contracts = max(1, int(notional / (premium * 100)))
    cost_entry  = n_contracts * premium * 100
    vega_s      = bs_vega(S_spot, K, T, r, vol_impl) * 2
    spread_cost = n_contracts * vega_s * bid_ask
    commission  = n_contracts * 0.65        # $0.65/contrat
    slippage    = cost_entry * 0.001        # 0.1%
    total_cost  = spread_cost + commission + slippage

    # ── Payoff à expiration
    expiry_row  = df_bt.iloc[min(i + HOLD_PERIOD, n_bt-1)]
    S_expiry    = expiry_row['SPX']
    intrinsic   = abs(S_expiry - K)
    payoff      = max(intrinsic, straddle_price(S_expiry, K, 1/252, r, expiry_row['RealizedVol']/100))

    if direction == 1:
        gross_pnl = n_contracts * (payoff - premium) * 100
    else:
        gross_pnl = n_contracts * (premium - payoff) * 100
        total_cost += spread_cost * 0.5   # coût de rachat

    net_pnl = gross_pnl - total_cost

    # ── Stop-loss
    net_pnl = max(net_pnl, -capital * STOP_LOSS)

    # ── Mise à jour
    capital += net_pnl
    capital  = max(capital, 0)

    # ── Vérification DD
    peak = max(capital_series)
    if peak > 0 and (capital - peak) / peak < -MAX_DD_LIMIT and not trading_halted:
        trading_halted = True
        print(f"  ⚠️  Trading arrêté le {dates_bt[i].date()} (DD={((capital-peak)/peak)*100:.1f}%)")

    capital_series.append(capital)
    trade_log.append({
        'date': dates_bt[i], 'direction': 'LONG' if direction==1 else 'SHORT',
        'signal': signal, 'vol_impl': row['VIX'], 'vol_pred': row['VolPred'],
        'S': S_spot, 'premium': premium, 'cost': total_cost,
        'gross_pnl': gross_pnl, 'net_pnl': net_pnl, 'capital': capital,
    })

while len(capital_series) < n_bt:
    capital_series.append(capital_series[-1])
capital_series = np.array(capital_series[:n_bt])
trades_df      = pd.DataFrame(trade_log)

# =============================================================================
# BLOC 6 — MÉTRIQUES
# =============================================================================
returns_daily = np.diff(capital_series) / np.maximum(capital_series[:-1], 1)
returns_daily = returns_daily[np.isfinite(returns_daily)]
total_ret     = (capital_series[-1] - CAPITAL_INIT) / CAPITAL_INIT
ann_ret       = (1 + total_ret) ** (252/n_bt) - 1
daily_std     = np.std(returns_daily)
sharpe        = (np.mean(returns_daily) / daily_std) * np.sqrt(252) if daily_std > 0 else 0
dd_series     = capital_series / np.maximum.accumulate(capital_series) - 1
max_dd        = np.min(dd_series)
calmar        = ann_ret / abs(max_dd) if max_dd != 0 else 0
n_trades      = len(trades_df)
win_rate      = (trades_df['net_pnl'] > 0).mean() if n_trades > 0 else 0
avg_win       = trades_df.loc[trades_df['net_pnl']>0,'net_pnl'].mean() if n_trades > 0 else 0
avg_loss      = trades_df.loc[trades_df['net_pnl']<0,'net_pnl'].mean() if n_trades > 0 else 0
profit_factor = abs(avg_win/avg_loss) if avg_loss != 0 else 0
total_costs   = trades_df['cost'].sum() if n_trades > 0 else 0

# Benchmark B&H SPX
spx_ret_bt = df_bt['SPX_ret'].values[:n_bt]
bh_capital = CAPITAL_INIT * np.exp(np.cumsum(spx_ret_bt))
bh_ret     = (bh_capital[-1] - CAPITAL_INIT) / CAPITAL_INIT
bh_dd      = bh_capital / np.maximum.accumulate(bh_capital) - 1
bh_sharpe  = (np.mean(spx_ret_bt)/np.std(spx_ret_bt))*np.sqrt(252)

print(f"\n{'='*60}")
print(f"RÉSULTATS INSTITUTIONNELS")
print(f"{'='*60}")
print(f"Capital initial  : ${CAPITAL_INIT:>12,.0f}")
print(f"Capital final    : ${capital_series[-1]:>12,.0f}")
print(f"Rendement total  : {total_ret*100:>+8.2f}%")
print(f"Rendement annuel : {ann_ret*100:>+8.2f}%")
print(f"Sharpe ratio     : {sharpe:>8.3f}")
print(f"Max drawdown     : {max_dd*100:>+8.2f}%")
print(f"Calmar ratio     : {calmar:>8.3f}")
print(f"Nb trades        : {n_trades:>8d}")
print(f"Win rate         : {win_rate*100:>8.1f}%")
print(f"Profit factor    : {profit_factor:>8.3f}")
print(f"Coûts totaux     : ${total_costs:>12,.0f}")
print(f"\nB&H SPX : Ret={bh_ret*100:+.2f}%  Sharpe={bh_sharpe:.3f}  MaxDD={np.min(bh_dd)*100:.1f}%")

# =============================================================================
# BLOC 7 — VISUALISATIONS
# =============================================================================
PLT_BG = '#F8FAFC'
fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor=PLT_BG)
fig.suptitle('Module 7 — Backtesting Institutionnel : Black-Scholes + Walk-Forward + Gestion du Risque',
             fontsize=12, fontweight='bold', color='#0D1117', y=1.01)

# ── Fig 1 : Capital ────────────────────────────
ax = axes[0, 0]
ax.plot(dates_bt[:n_bt], capital_series/1e6, color='#7C3AED', lw=2,
        label=f'Stratégie Vol  Sharpe={sharpe:.2f}')
ax.plot(dates_bt[:n_bt], bh_capital/1e6, color='#374151', lw=1.5,
        linestyle='--', alpha=0.6, label=f'B&H SPX  Sharpe={bh_sharpe:.2f}')
ax.axhline(CAPITAL_INIT/1e6, color='#6B7280', lw=0.8, linestyle=':', alpha=0.5)
for rd in retrain_dates:
    if rd >= dates_bt[0]:
        ax.axvline(pd.Timestamp(rd), color='#F59E0B', lw=0.5, alpha=0.4)
ax.set_ylabel('Capital ($M)', fontsize=9)
ax.set_title('Évolution du Capital ($M)', fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.set_facecolor(PLT_BG)
ax.grid(True, alpha=0.3, linestyle='--'); ax.spines[['top','right']].set_visible(False)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ── Fig 2 : Drawdowns ─────────────────────────
ax = axes[0, 1]
ax.fill_between(dates_bt[:n_bt], dd_series*100, 0, alpha=0.4, color='#7C3AED')
ax.plot(dates_bt[:n_bt], dd_series*100, color='#7C3AED', lw=1.2,
        label=f'Stratégie  Max={max_dd*100:.1f}%')
ax.plot(dates_bt[:n_bt], bh_dd[:n_bt]*100, color='#374151', lw=1, linestyle='--',
        alpha=0.6, label=f'B&H SPX  Max={np.min(bh_dd)*100:.1f}%')
ax.axhline(-MAX_DD_LIMIT*100, color='#DC2626', lw=1.5, linestyle='--',
           label=f'Limite DD ({-MAX_DD_LIMIT*100:.0f}%)')
ax.set_ylabel('Drawdown (%)', fontsize=9)
ax.set_title('Drawdowns — Limite automatique d\'arrêt', fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.set_facecolor(PLT_BG)
ax.grid(True, alpha=0.3, linestyle='--'); ax.spines[['top','right']].set_visible(False)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ── Fig 3 : Distribution P&L ──────────────────
ax = axes[1, 0]
if n_trades > 0:
    pnl_arr = trades_df['net_pnl'].values
    ax.hist(pnl_arr[pnl_arr>0]/1000, bins=20, color='#10B981', alpha=0.7,
            label=f'Gagnants ({(pnl_arr>0).sum()})', edgecolor='white')
    ax.hist(pnl_arr[pnl_arr<0]/1000, bins=20, color='#EF4444', alpha=0.7,
            label=f'Perdants ({(pnl_arr<0).sum()})', edgecolor='white')
    ax.axvline(0, color='#374151', lw=1.5)
    ax.axvline(pnl_arr.mean()/1000, color='#F59E0B', lw=2, linestyle='--',
               label=f'Moy = ${pnl_arr.mean()/1000:.1f}k')
ax.set_xlabel('P&L net par trade ($k)', fontsize=9)
ax.set_title('Distribution des P&L par Trade', fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.set_facecolor(PLT_BG)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.spines[['top','right']].set_visible(False)

# ── Fig 4 : Prix straddle vs VIX ──────────────
ax = axes[1, 1]
vix_range = np.linspace(10, 65, 200)
S_demo, r_demo = 5000, 0.05
for T_days, col, lbl in [(7,'#EF4444','7j'),(21,'#F59E0B','21j'),
                          (30,'#10B981','30j'),(60,'#2563EB','60j')]:
    prices = [straddle_price(S_demo, S_demo, T_days/252, r_demo, v/100)/S_demo*100
              for v in vix_range]
    ax.plot(vix_range, prices, lw=2, label=f'T={lbl}', color=col)
ax.axvline(20, color='#6B7280', lw=1, linestyle='--', alpha=0.5)
ax.set_xlabel('VIX (%)', fontsize=9)
ax.set_ylabel('Prix straddle (% du spot)', fontsize=9)
ax.set_title('Prix du Straddle ATM vs VIX\n(SPX=5000, T=horizon)', fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.set_facecolor(PLT_BG)
ax.grid(True, alpha=0.3, linestyle='--'); ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('module7_institutional_backtest.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGraphique sauvegardé : module7_institutional_backtest.png")

# =============================================================================
# BLOC 8 — PRIX D'OPTIONS ACTUELS (yfinance)
# =============================================================================
print("\n" + "="*60)
print("PRIX D'OPTIONS SPX ACTUELS (30 jours)")
print("="*60)
try:
    spx_ticker = yf.Ticker("^SPX")
    expirations = spx_ticker.options
    # Trouver l'expiration la plus proche de 30 jours
    from datetime import datetime, timedelta
    target = datetime.now() + timedelta(days=30)
    exp_dates = [datetime.strptime(e, '%Y-%m-%d') for e in expirations]
    closest   = min(exp_dates, key=lambda d: abs((d - target).days))
    exp_str   = closest.strftime('%Y-%m-%d')
    chain     = spx_ticker.option_chain(exp_str)
    calls     = chain.calls
    puts      = chain.puts

    # Trouver les options ATM
    S_current = spx_ticker.history(period='1d')['Close'].iloc[-1]
    atm_call  = calls.iloc[(calls['strike'] - S_current).abs().argsort()[:1]]
    atm_put   = puts.iloc[(puts['strike']  - S_current).abs().argsort()[:1]]

    print(f"SPX actuel    : ${S_current:,.2f}")
    print(f"Expiration    : {exp_str}")
    print(f"\nATM Call:")
    print(f"  Strike      : ${atm_call['strike'].values[0]:,.0f}")
    print(f"  Bid/Ask     : ${atm_call['bid'].values[0]:.2f} / ${atm_call['ask'].values[0]:.2f}")
    print(f"  Vol implicite: {atm_call['impliedVolatility'].values[0]*100:.1f}%")
    print(f"\nATM Put:")
    print(f"  Strike      : ${atm_put['strike'].values[0]:,.0f}")
    print(f"  Bid/Ask     : ${atm_put['bid'].values[0]:.2f} / ${atm_put['ask'].values[0]:.2f}")
    print(f"  Vol implicite: {atm_put['impliedVolatility'].values[0]*100:.1f}%")

    straddle_mid = ((atm_call['bid'].values[0] + atm_call['ask'].values[0])/2 +
                    (atm_put['bid'].values[0]  + atm_put['ask'].values[0])/2)
    straddle_pct = straddle_mid / S_current * 100
    print(f"\nStraddle ATM  : ${straddle_mid:.2f} ({straddle_pct:.2f}% du spot)")

    # Vol implicite vs vol prédite si modèle disponible
    vix_now = yf.Ticker("^VIX").history(period='1d')['Close'].iloc[-1]
    print(f"\nVIX actuel    : {vix_now:.2f}")
    print(f"Signal potentiel : vol_pred - VIX = ??? (lancer le modèle sur données récentes)")

except Exception as e:
    print(f"Données options non disponibles : {e}")
    print("(Utiliser un compte premium Yahoo Finance ou OptionMetrics pour les données réelles)")
