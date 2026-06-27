# Fund Analysis Engine

**Interactive Streamlit app for quantitative fund evaluation using TOPSIS and Yuan & Yuan (2023)**

Qaweem M Ahmad · MSc Finance · University of Nottingham · BUSI4519 · 2025

---

## Live Demo

**[fund-analysis-engine.streamlit.app](https://fund-analysis-engine.streamlit.app)**

---

## Overview

An interactive web application that evaluates actively managed emerging market equity funds against the MSCI Emerging Markets Index. Users upload return data, configure pillar weights, and receive ranked results across 19 financial metrics using two independent multi-criteria decision analysis (MCDA) methods.

Built as part of the Capital Markets Analysis coursework (BUSI4519), with from-scratch Python implementations of TOPSIS and the Yuan & Yuan (2023) paired competition eigenvector method.

---

## Key Features

- Upload CSV return data and run a full analysis in one click
- 19 financial metrics computed from monthly log returns, no third-party analytics libraries
- Two independent ranking models: TOPSIS and Yuan & Yuan (2023)
- Adjustable pillar weights with live sensitivity analysis
- Portfolio optimisation: min-variance, max-Sharpe, equal-weight, risk-parity
- Bootstrap confidence intervals for all metrics
- Downloadable Excel report and ranking summary
- Interactive Plotly charts throughout
- Executive Summary page consolidating top-line results

---

## Metrics Covered (19)

| Pillar | Metrics |
|---|---|
| Returns (40%) | Annualised Return, Alpha, Upside Capture, Calmar Ratio |
| Risk-Adjusted (25%) | Sharpe, Sortino, Treynor, Information Ratio, R² |
| Risk & Drawdown (20%) | Annualised Volatility, Beta, Tracking Error, Max Drawdown, Max DD Duration, Downside Capture |
| Costs (10%) | OCF |
| ESG (5%) | ESG Globe Rating, Carbon Risk Score |

Default pillar weights shown above; all weights are user-adjustable via sliders.

---

## Ranking Methods

### TOPSIS (Hwang & Yoon, 1981)

Five-step methodology:

1. Min-max normalisation to [0, 1]
2. Pillar-weighted matrix
3. Positive Ideal Solution (PIS) and Negative Ideal Solution (NIS)
4. Euclidean distances from PIS and NIS
5. Relative closeness score: $S_i = D_i^- / (D_i^+ + D_i^-)$

### Yuan & Yuan (2023) Paired Competition

1. Competition matrix: for each fund pair (i, j), the weighted proportion of metrics on which fund i outperforms fund j
2. Principal eigenvector of the competition matrix via power iteration (typically converges in under 30 steps)
3. Scores normalised to sum to 1 and ranked

Both models share the same pillar weights and metric set, making results directly comparable.

---

## Portfolio and Reporting Features

- **Portfolio Optimisation**: min-variance, max-Sharpe, equal-weight, and risk-parity allocations with efficient frontier chart
- **Bootstrap CI**: 95% confidence intervals for all 19 metrics via resampling (default 1,000 iterations)
- **Sensitivity Analysis**: rankings re-computed under four alternative weighting schemes (Return-Heavy, Risk-Heavy, Equal, Baseline)
- **Excel Report**: one-click download of all metrics, rankings, and portfolio weights
- **Naive Ranking**: simple composite baseline for sanity-checking MCDA results

---

## Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/Qaweem-Ahmad/Fund-Analysis-Engine-Website.git
cd Fund-Analysis-Engine-Website
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. An internet connection is not required after install; return data is uploaded via CSV.

---

## Project Structure

```
Fund-Analysis-Engine-Website/
|
|-- app.py                  # Main Streamlit application (~4,900 lines)
|-- requirements.txt
|-- .gitignore
|-- README.md
|
|-- modules/
|   |-- topsis.py           # TOPSIS class (normalise, weight, PIS/NIS, score)
|   |-- yuan.py             # YuanYuan class (competition matrix, power iteration)
|   |-- naive_ranking.py    # Simple composite baseline ranking
|   `-- portfolio.py        # Portfolio optimisation and efficient frontier
|
|-- tests/
|   |-- conftest.py         # Shared fixtures
|   |-- test_core_metrics.py
|   |-- test_downside_metrics.py
|   |-- test_rankings.py
|   |-- test_topsis.py
|   |-- test_yuan.py
|   |-- test_table_highlighting.py
|   |-- test_summary_cards.py
|   `-- test_silent_fallbacks.py
|
`-- .streamlit/
    `-- config.toml         # Theme: light, primary colour #154D57
```

---

## Tests

```bash
pytest tests/
```

Current status: **98 passed, 2 xfailed** (expected failures for edge-case placeholders).

---

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) from the `main` branch of this repository. No additional configuration required beyond `requirements.txt`.

---

## Academic Context

- Hwang, C-L. and Yoon, K. (1981) *Multiple Attribute Decision Making*. Springer-Verlag.
- Yuan, J. and Yuan, X. (2023) 'A Comprehensive Method for Ranking Mutual Fund Performance', *SAGE Open*, 13(2).
- Sharpe, W.F. (1964) 'Capital Asset Prices', *Journal of Finance*, 19(3).
- Sortino, F.A. and Satchell, S. (2001) *Managing Downside Risk in Financial Markets*. Butterworth-Heinemann.

---

## Limitations

- ETF proxies substitute for UK-domiciled OEIC NAV data (not available on Yahoo Finance)
- Sample window January 2020 to October 2025 does not represent a full EM market cycle
- With n=4 funds, results are illustrative rather than statistically conclusive

---

*Capital Markets Analysis · BUSI4519 · University of Nottingham Business School*
