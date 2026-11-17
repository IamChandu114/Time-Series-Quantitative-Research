"""
Time-series research starter: stationarity tests, autocorrelation tests, AR modelling,
and a leakage-free walk-forward momentum backtest with costs and a bootstrap test.

Usage:
  python research.py --csv prices.csv            # columns: Date, Close (daily prices)
  python research.py --synthetic random_walk     # sanity check: should find NO edge
  python research.py --synthetic ar_edge         # sanity check: should detect the planted edge

Only numpy / pandas / scipy / matplotlib are required.
Everything is seeded, so results are reproducible. Report what you actually measure.
"""
import argparse, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANN = 252
SEED = 7

# ---------------------------------------------------------------- data
def load_csv(path):
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    return df.set_index("Date")["Close"].astype(float).dropna()

def synthetic(kind, n=3000, seed=SEED):
    rng = np.random.default_rng(seed)
    if kind == "random_walk":
        r = rng.normal(0.0003, 0.01, n)                 # no predictability
    elif kind == "ar_edge":
        r = np.zeros(n)                                 # planted positive autocorrelation
        e = rng.normal(0.0, 0.01, n)
        for t in range(1, n):
            r[t] = 0.08 * r[t - 1] + e[t]
    else:
        raise ValueError(kind)
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)

# ---------------------------------------------------------------- statistics
def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), resid

def adf(x, lags=1):
    """Augmented Dickey-Fuller (constant, `lags` lagged differences). Returns (stat, verdict)."""
    x = np.asarray(x); dx = np.diff(x)
    y = dx[lags:]
    cols = [x[lags:-1], np.ones(len(y))] + [dx[lags - k - 1: len(dx) - k - 1] for k in range(lags)]
    X = np.column_stack(cols)
    beta, se, _ = ols(y, X)
    tstat = beta[0] / se[0]
    crit = {"1%": -3.43, "5%": -2.86, "10%": -2.57}   # asymptotic, constant case
    verdict = "stationary (reject unit root at 5%)" if tstat < crit["5%"] else "unit root not rejected at 5%"
    return float(tstat), verdict

def ljung_box(x, lags=10):
    x = np.asarray(x) - np.mean(x); n = len(x)
    acf = np.array([np.sum(x[k:] * x[:n - k]) for k in range(lags + 1)]) / np.sum(x * x)
    q = n * (n + 2) * np.sum(acf[1:] ** 2 / (n - np.arange(1, lags + 1)))
    return float(q), float(1 - stats.chi2.cdf(q, lags)), acf[1:]

def fit_ar(r, max_p=5):
    """AR(p) by OLS, choose p by AIC."""
    best = None
    for p in range(1, max_p + 1):
        y = r[p:]
        X = np.column_stack([np.ones(len(y))] + [r[p - k - 1: len(r) - k - 1] for k in range(p)])
        beta, se, resid = ols(y, X)
        n = len(y); aic = n * np.log(resid @ resid / n) + 2 * (p + 1)
        if best is None or aic < best["aic"]:
            best = {"p": p, "aic": float(aic), "beta": beta.tolist(), "t": (beta / se).tolist()}
    return best

# ---------------------------------------------------------------- strategy
def positions(ret, lookback):
    """Sign of trailing mean return, shifted one bar -> uses ONLY past data (no lookahead)."""
    sig = np.sign(ret.rolling(lookback).mean())
    return sig.shift(1).fillna(0.0)

def pnl(ret, pos, cost_bps):
    turn = pos.diff().abs().fillna(pos.abs())
    return pos * ret - turn * cost_bps / 1e4

def metrics(x):
    x = pd.Series(x).dropna()
    mu, sd = x.mean() * ANN, x.std() * np.sqrt(ANN)
    eq = (1 + x).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return {"ann_return": float(mu), "ann_vol": float(sd),
            "sharpe": float(mu / sd) if sd > 0 else 0.0, "max_drawdown": float(dd)}

def block_bootstrap_sharpe(x, n_boot=2000, block=10, seed=SEED):
    rng = np.random.default_rng(seed); x = np.asarray(x); n = len(x)
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, size=int(np.ceil(n / block)))
        s = np.concatenate([x[i:i + block] for i in starts])[:n]
        out.append(s.mean() / s.std() * np.sqrt(ANN) if s.std() > 0 else 0.0)
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--synthetic", choices=["random_walk", "ar_edge"])
    ap.add_argument("--cost_bps", type=float, default=5.0)
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    price = load_csv(a.csv) if a.csv else synthetic(a.synthetic or "random_walk")
    ret = np.log(price).diff().dropna()
    R = {"n_obs": int(len(ret))}

    # 1) stationarity + autocorrelation
    R["adf_price"] = adf(np.log(price.values))
    R["adf_returns"] = adf(ret.values)
    q, p, acf = ljung_box(ret.values, 10)
    R["ljung_box_returns"] = {"Q": q, "p_value": p, "acf_lag1": float(acf[0])}
    R["ar_fit"] = fit_ar(ret.values)

    # 2) walk-forward: choose lookback in-sample, evaluate untouched out-of-sample
    split = int(len(ret) * a.train_frac)
    tr, te = ret.iloc[:split], ret.iloc[split:]
    grid = [1, 2, 5, 10, 20, 40, 60, 120]
    is_sharpe = {L: metrics(pnl(tr, positions(tr, L), a.cost_bps))["sharpe"] for L in grid}
    best_L = max(is_sharpe, key=is_sharpe.get)
    # positions for the test window use history from the full series (still past-only)
    pos_full = positions(ret, best_L)
    oos = pnl(te, pos_full.loc[te.index], a.cost_bps)
    R["selected_lookback"] = int(best_L)
    R["in_sample_sharpe_by_lookback"] = {str(k): float(v) for k, v in is_sharpe.items()}
    R["oos_strategy"] = metrics(oos)
    R["oos_buy_and_hold"] = metrics(te)
    lo, hi, p_le0 = block_bootstrap_sharpe(oos.values)
    t, tp = stats.ttest_1samp(oos.values, 0.0)
    R["oos_sharpe_95ci_block_bootstrap"] = [lo, hi]
    R["oos_bootstrap_p_sharpe_le_0"] = p_le0
    R["oos_mean_return_ttest"] = {"t": float(t), "p_two_sided": float(tp)}
    R["settings"] = {"cost_bps_per_position_change": a.cost_bps, "train_frac": a.train_frac, "seed": SEED}

    json.dump(R, open(f"{a.out}.json", "w"), indent=2)
    fig, ax = plt.subplots(figsize=(8, 4))
    (1 + oos).cumprod().plot(ax=ax, label=f"strategy OOS (L={best_L})")
    (1 + te).cumprod().plot(ax=ax, label="buy & hold OOS")
    ax.legend(); ax.set_title("Out-of-sample equity (net of costs)"); fig.tight_layout(); fig.savefig(f"{a.out}.png", dpi=130)

    print(json.dumps(R, indent=2))
    print("\nREAD THIS HONESTLY: one asset, one strategy family, one split. A positive OOS Sharpe here is a "
          "hypothesis, not proof. Check multiple assets/periods and report the failures too.")

if __name__ == "__main__":
    main()
