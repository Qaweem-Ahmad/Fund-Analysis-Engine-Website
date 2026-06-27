import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict


def compute_core_metrics(fund_returns: pd.Series, benchmark_returns: pd.Series, rf_annual: float = 0.05) -> Dict[str, float]:
    """
    Compute core performance metrics for a fund relative to benchmark.
    """
    rf_monthly = rf_annual / 12
    excess_returns = fund_returns - rf_monthly
    active_returns = fund_returns - benchmark_returns

    # Annualized Return
    ann_return = (np.exp(fund_returns.mean() * 12) - 1) * 100

    # Annualized Volatility
    ann_vol = fund_returns.std() * np.sqrt(12) * 100

    # Alpha and Beta from OLS
    slope, intercept, r_value, p_value, std_err = linregress(benchmark_returns - rf_monthly, fund_returns - rf_monthly)
    alpha_ann = (np.exp(intercept * 12) - 1) * 100
    beta = slope
    r_squared = r_value ** 2

    # Sharpe Ratio — geometric annualisation consistent with ann_return
    sharpe = ((np.exp(fund_returns.mean() * 12) - 1) - rf_annual) / (fund_returns.std() * np.sqrt(12)) if fund_returns.std() > 0 else np.nan

    # Treynor Ratio — geometric annualisation consistent with ann_return
    treynor = ((np.exp(fund_returns.mean() * 12) - 1) - rf_annual) / beta if beta != 0 else np.nan

    # Sortino Ratio — downside_dev is monthly std, must annualise denominator by sqrt(12)
    downside_returns = excess_returns[excess_returns < 0]
    downside_dev = downside_returns.std() if len(downside_returns) > 0 else np.nan
    sortino = (excess_returns.mean() * np.sqrt(12)) / downside_dev if downside_dev and downside_dev > 0 else np.nan

    # Information Ratio
    tracking_error = active_returns.std() * np.sqrt(12)
    info_ratio = (active_returns.mean() * 12) / tracking_error if tracking_error > 0 else np.nan
    tracking_error_pct = tracking_error * 100

    return {
        'Ann. Return (%)': ann_return,
        'Ann. Volatility (%)': ann_vol,
        'Alpha (ann. %)': alpha_ann,
        'Beta': beta,
        'R²': r_squared,
        'Sharpe Ratio': sharpe,
        'Treynor Ratio': treynor,
        'Sortino Ratio': sortino,
        'Information Ratio': info_ratio,
        'Tracking Error (%)': tracking_error_pct,
    }


def compute_downside_metrics(fund_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    """
    Compute downside risk metrics for a fund relative to benchmark.
    """
    # Upside and Downside Capture
    up_months = benchmark_returns > 0
    down_months = benchmark_returns < 0

    upside_capture = (fund_returns[up_months].mean() / benchmark_returns[up_months].mean()) * 100 if benchmark_returns[up_months].mean() != 0 else np.nan
    downside_capture = (fund_returns[down_months].mean() / benchmark_returns[down_months].mean()) * 100 if benchmark_returns[down_months].mean() != 0 else np.nan

    # Max Drawdown
    wealth = np.exp(fund_returns.cumsum())
    cummax = wealth.cummax()
    drawdown = (wealth - cummax) / cummax
    max_drawdown = drawdown.min() * 100

    # Max DD Duration
    below_peak = wealth < cummax
    if below_peak.any():
        # Find consecutive True sequences
        groups = below_peak.ne(below_peak.shift()).cumsum()
        durations = below_peak.groupby(groups).sum()
        max_dd_duration = durations.max()
    else:
        max_dd_duration = 0

    # Calmar Ratio
    ann_return = (np.exp(fund_returns.mean() * 12) - 1)
    calmar = ann_return / abs(max_drawdown / 100) if max_drawdown != 0 else np.nan

    return {
        'Upside Capture (%)': upside_capture,
        'Downside Capture (%)': downside_capture,
        'Max Drawdown (%)': max_drawdown,
        'Max DD Duration (mths)': max_dd_duration,
        'Calmar Ratio': calmar,
    }