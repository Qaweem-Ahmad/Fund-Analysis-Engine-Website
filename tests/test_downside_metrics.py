"""Known-answer unit tests for downside metric formulas in modules/metrics.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest
from modules.metrics import compute_downside_metrics


# ── Manual reference implementations ─────────────────────────────────────────

def _manual_max_drawdown(r):
    wealth = np.exp(r.cumsum())
    cummax = wealth.cummax()
    dd = (wealth - cummax) / cummax
    return dd.min() * 100

def _manual_calmar(r, max_dd_pct):
    ann_ret = (np.exp(r.mean() * 12) - 1)
    return ann_ret / abs(max_dd_pct / 100)

def _manual_upside_capture(fund_r, bm_r):
    up = bm_r > 0
    return (fund_r[up].mean() / bm_r[up].mean()) * 100

def _manual_downside_capture(fund_r, bm_r):
    down = bm_r < 0
    return (fund_r[down].mean() / bm_r[down].mean()) * 100


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMaxDrawdown:
    def test_known_value(self, fund_returns, benchmark_returns):
        result = compute_downside_metrics(fund_returns, benchmark_returns)
        expected = _manual_max_drawdown(fund_returns)
        assert abs(result["Max Drawdown (%)"] - expected) < 1e-10

    def test_is_negative(self, fund_returns, benchmark_returns):
        """Max drawdown must always be <= 0."""
        result = compute_downside_metrics(fund_returns, benchmark_returns)
        assert result["Max Drawdown (%)"] <= 0

    def test_consistently_negative_stored_as_negative(self, fund_returns, benchmark_returns):
        """
        Regression: drawdown values must be stored as negative numbers.
        Less negative = better preservation. idxmax() gives best drawdown.
        """
        result = compute_downside_metrics(fund_returns, benchmark_returns)
        dd = result["Max Drawdown (%)"]
        assert dd < 0, f"Max drawdown stored as {dd}, expected a negative number"

    def test_flat_returns_give_zero_drawdown(self, flat_returns, benchmark_returns):
        result = compute_downside_metrics(flat_returns, benchmark_returns)
        assert abs(result["Max Drawdown (%)"]) < 1e-10

    def test_positive_only_returns_give_zero_drawdown(self, benchmark_returns):
        """Monotonically increasing wealth path has no drawdown."""
        strictly_up = pd.Series(np.full(60, 0.01))
        result = compute_downside_metrics(strictly_up, benchmark_returns)
        assert abs(result["Max Drawdown (%)"]) < 1e-10

    def test_less_negative_is_better(self, benchmark_returns):
        """Fund with smaller losses should have higher (less negative) max drawdown."""
        rng = np.random.default_rng(1)
        volatile = pd.Series(rng.normal(0, 0.08, 60))
        stable   = pd.Series(rng.normal(0, 0.02, 60))
        r_vol = compute_downside_metrics(volatile, benchmark_returns)["Max Drawdown (%)"]
        r_stb = compute_downside_metrics(stable,   benchmark_returns)["Max Drawdown (%)"]
        assert r_stb > r_vol, "More stable fund should have less negative max drawdown"


class TestMaxDrawdownDuration:
    def test_flat_returns_give_zero_duration(self, flat_returns, benchmark_returns):
        result = compute_downside_metrics(flat_returns, benchmark_returns)
        assert result["Max DD Duration (mths)"] == 0

    def test_duration_is_non_negative(self, fund_returns, benchmark_returns):
        result = compute_downside_metrics(fund_returns, benchmark_returns)
        assert result["Max DD Duration (mths)"] >= 0

    def test_known_value(self, fund_returns, benchmark_returns):
        result = compute_downside_metrics(fund_returns, benchmark_returns)
        # Manual: count longest run below running peak
        wealth = np.exp(fund_returns.cumsum())
        cummax = wealth.cummax()
        below = wealth < cummax
        groups = below.ne(below.shift()).cumsum()
        durations = below.groupby(groups).sum()
        expected = durations.max()
        assert result["Max DD Duration (mths)"] == expected


class TestCaptureRatios:
    def test_upside_capture_known_value(self, fund_returns, benchmark_returns):
        result = compute_downside_metrics(fund_returns, benchmark_returns)
        expected = _manual_upside_capture(fund_returns, benchmark_returns)
        assert abs(result["Upside Capture (%)"] - expected) < 1e-10

    def test_downside_capture_known_value(self, fund_returns, benchmark_returns):
        result = compute_downside_metrics(fund_returns, benchmark_returns)
        expected = _manual_downside_capture(fund_returns, benchmark_returns)
        assert abs(result["Downside Capture (%)"] - expected) < 1e-10

    def test_benchmark_vs_itself_upside_is_100(self, benchmark_returns):
        result = compute_downside_metrics(benchmark_returns, benchmark_returns)
        assert abs(result["Upside Capture (%)"] - 100.0) < 1e-8

    def test_benchmark_vs_itself_downside_is_100(self, benchmark_returns):
        result = compute_downside_metrics(benchmark_returns, benchmark_returns)
        assert abs(result["Downside Capture (%)"] - 100.0) < 1e-8


class TestCalmarRatio:
    def test_known_value(self, fund_returns, benchmark_returns):
        result = compute_downside_metrics(fund_returns, benchmark_returns)
        dd = result["Max Drawdown (%)"]
        expected = _manual_calmar(fund_returns, dd)
        assert abs(result["Calmar Ratio"] - expected) < 1e-10

    def test_zero_drawdown_gives_nan(self, flat_returns, benchmark_returns):
        """Zero drawdown means division by zero — Calmar should be NaN."""
        result = compute_downside_metrics(flat_returns, benchmark_returns)
        assert np.isnan(result["Calmar Ratio"])
