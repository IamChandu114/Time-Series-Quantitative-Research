# Time-series research starter

One script, `research.py`, that does what a first quant-research write-up needs:
1. Stationarity (ADF) on log-prices vs. returns; Ljung-Box autocorrelation test; AR(p) fit chosen by AIC.
2. A leakage-free walk-forward momentum backtest: pick the lookback on the first 60% of data, evaluate on the untouched last 40%, net of transaction costs.
3. Significance: block-bootstrap 95% CI for the out-of-sample Sharpe and a t-test on mean returns.

Sanity checks (run first): `--synthetic random_walk` should find NO edge; `--synthetic ar_edge` should detect the planted lag-1 autocorrelation (Ljung-Box, AR fit, and a positive out-of-sample Sharpe, which shrinks as costs rise; try `--cost_bps 1` vs `5`).

Real run: download daily prices (e.g. an index or a few liquid stocks) to `prices.csv` with columns `Date,Close`, then
`python research.py --csv prices.csv`. Repeat on several assets and periods.

For a 9+ level write-up, extend it: multiple assets, rolling-window re-fits, other signals (mean reversion, volatility),
a multiple-testing correction, and a short README stating what failed. Never tune on the test window.
