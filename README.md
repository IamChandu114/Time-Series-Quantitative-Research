
# Time-Series Quantitative Research

Statistical analysis and out-of-sample evaluation of financial time-series strategies using Python.

## Research Objectives

- Test price and return stationarity using Augmented Dickey-Fuller (ADF).
- Examine return autocorrelation using Ljung-Box diagnostics.
- Fit autoregressive models.
- Evaluate trading signals using walk-forward backtesting.
- Account for transaction costs.
- Quantify uncertainty using block-bootstrap confidence intervals.
- Compare strategy performance against buy-and-hold.

## Technology Stack

Python | NumPy | Pandas | SciPy | Matplotlib | Statsmodels

## Initial Empirical Study: SPY

Dataset: SPY adjusted daily prices, 2010–2026.

The initial experiment found:

- Prices were non-stationary.
- Returns were stationary.
- Lag-1 return autocorrelation was approximately -0.10.
- AR(1) coefficient t-statistic was approximately -6.6.

### Out-of-Sample Results

| Metric | Momentum Strategy | Buy & Hold |
|---|---:|---:|
| Sharpe Ratio | -0.07 | 0.65 |
| Bootstrap 95% CI | [-0.72, 0.64] | — |

Transaction costs: 5 bps per position change.

The tested momentum strategy did not demonstrate a statistically
reliable out-of-sample edge.

## Research Limitations

This is an initial study involving one asset and one strategy family.
The findings are not evidence of a persistent or exploitable trading edge.

Further work should evaluate multiple assets, market regimes,
strategy families and transaction-cost assumptions.

## Reproducibility

Clone the repository:

```bash
git clone https://github.com/IamChandu114/Time-Series-Quantitative-Research.git
cd Time-Series-Quantitative-Research
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run synthetic sanity checks:

```bash
python research.py --synthetic random_walk
python research.py --synthetic ar_edge
```

Run the SPY analysis:

```bash
python research.py --csv prices.csv --out spy
```

## Author

Chandu Vemula

GitHub: https://github.com/IamChandu114

LinkedIn: https://linkedin.com/in/pace1304
