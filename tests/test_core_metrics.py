"""Known-answer unit tests for core metric formulas in modules/metrics.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest
from scipy.stats import linregress
from modules.metrics import compute_core_metrics


# ── Helpers for manual reference values ──────────────────────────────────────

def _manual_ann_return(r):
    return (np.exp(r.mean() * 12) - 1) * 100

def _manual_ann_vol(r):
    return r.std() * np.sqrt(12) * 100

def _manual_alpha_beta_r2(fund_r, bm_r, rf_m):
    slope, intercept, r_val, *_ = linregress(bm_r - rf_m, fund_r - rf_m)
    alpha = (np.exp(intercept * 12) - 1) * 100
    return alpha, slope, r_val ** 2

def _manual_sharpe(r, rf_annual):
    ann_ret = (np.exp(r.mean() * 12) - 1)
    return (ann_ret - rf_annual) / (r.std() * np.sqrt(12))

def _manual_treynor(r, beta, rf_annual):
    ann_ret = (np.exp(r.mean() * 12) - 1)
    return (ann_ret - rf_annual) / beta

def _manual_sortino(r, rf_monthly):
    excess = r - rf_monthly
    neg = excess[excess < 0]
    dd = neg.std()
    return (excess.mean() * np.sqrt(12)) / dd

def _manual_info_ratio(fund_r, bm_r):
    active = fund_r - bm_r
    te = active.std() * np.sqrt(12)
    return (active.mean() * 12) / te


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAnnualisedReturn:
    def test_known_value(self, fund_returns, benchmark_returns, rf_annual):
        result = compute_core_metrics(fund_returns, benchmark_returns, rf_annual)
        expected = _manual_ann_return(fund_returns)
        assert abs(result["Ann. Return (%)"] - expected) < 1e-10

    def test_zero_returns_gives_zero(self, flat_returns, benchmark_returns, rf_annual):
        result = compute_core_metrics(flat_returns, benchmark_returns, rf_annual)
        assert abs(result["Ann. Return (%)"]) < 1e-10

    def test_positive_drift_gives_positive(self, fund_returns, benchmark_returns, rf_annual):
        result = compute_core_metrics(fund_returns, benchmark_returns, rf_annual)
        assert result["Ann. Return (%)"] > 0


class TestAnnualisedVolatility:
    def test_known_value(self, fund_returns, benchmark_returns, rf_annual):
        result = compute_core_metrics(fund_returns, benchmark_returns, rf_annual)
        expected = _manual_ann_vol(fund_returns)
        assert abs(result["Ann. Volatility (%)"] - expected) < 1e-10

    def test_flat_returns_give_zero_vol(self, flat_returns, benchmark_returns, rf_annual):
        result = compute_core_metrics(flat_returns, benchmark_returns, rf_annual)
        assert result["Ann. Volatility (%)"] == 0.0


class TestAlphaBetaR2:
    def test_beta_known_value(self, fund_returns, benchmark_returns, rf_annual, rf_monthly):
        result = compute_core_metrics(fund_returns, benchmark_returns, rf_annual)
        _, expected_beta, _ = _manual_alpha_beta_r2(fund_returns, benchmark_returns, rf_monthly)
        assert abs(result["Beta"] - expected_beta) < 1e-10

    def test_alpha_known_value(self, fund_returns, benchmark_returns, rf_annual, rf_monthly):
        result = compute_core_metrics(fund_returns, benchmark_returns, rf_annual)
        expected_alpha, _, _ = _manual_alpha_beta_r2(fund_returns, benchmark_returns, rf_monthly)
        assert abs(result["Alpha (ann. %)"] - expected_alpha) < 1e-10

    def test_r2_known_value(self, fund_returns, benchmark_returns, rf_annual, rf_monthly):
        result = compute_core_metrics(fund_returns, benchmark_returns, rf_annual)
        _, _, expected_r2 = _manual_alpha_beta_r2(fund_returns, benchmark_returns, rf_monthly)
        assert abs(result["R²"] - expected_r2) < 1e-10

    def test_benchmark_vs_itself_beta_is_1(self, benchmark_returns, rf_annual):
        result = compute_core_metrics(benchmark_returns, benchmark_returns, rf_annual)
        assert abs(result["Beta"] - 1.0) < 1e-10

    def test_benchmark_vs_itself_r2_is_1(self, benchmark_returns, rf_annual):
        result = compute_core_metrics(benchmark_returns, benchmark_returns, rf_annual)
        assert abs(result["R²"] - 1.0) < 1e-10

    def test_benchmark_vs_itself_alpha_is_0(self, benchmark_returns, rf_annual):
        """Alpha of benchmark against itself must be exactly 0."""
        result = compute_core_metrics(benchmark_returns, benchmark_returns, rf_annual)
        assert abs(result["Alpha (ann. %)"]) < 1e-8


class TestSharpeRatio:
    def test_known_value(self, fund_returns, benchmark_returns, rf_annual):
        result = compute_core_metrics(fund_returns, benchmark_returns, rf_annual)
        expected = _manual_sharpe(fund_returns, rf_annual)
        assert abs(result["Sharpe Ratio"] - expected) < 1e-10

    def test_zero_vol_gives_nan(self, flat_returns, benchmark_returns, rf_annual):
        result = compute_core_metrics(flat_returns, benchmark_returns, rf_annual)
        assert np.isnan(result["Sharpe Ratio"])


class TestTreynorRatio:
    def test_known_value(self, fund_returns, benchmark_returns, rf_annual, rf_monthly):
        result = compute_core_metrics(fund_returns, benchmark_returns, rf_annual)
        _, beta, _ = _manual_alpha_beta_r2(fund_returns, benchmark_returns, rf_monthly)
        expected = _manual_treynor(fund_returns, beta, rf_annual)
        assert abs(result["Treynor Ratio"] - expected) < 1e-10


class TestSortinoRatio:
    def test_known_value(self, fund_returns, benchmark_returns, rf_annual, rf_monthly):
        result = compute_core_metrics(fund_returns, benchmark_returns, rf_annual)
        expected = _manual_sortino(fund_returns, rf_monthly)
        assert abs(result["Sortino Ratio"] - expected) < 1e-10


class TestInformationRatio:
    def test_known_value(self, fund_returns, benchmark_returns, rf_annual):
        result = compute_core_metrics(fund_returns, benchmark_returns, rf_annual)
        expected = _manual_info_ratio(fund_returns, benchmark_returns)
        assert abs(result["Information Ratio"] - expected) < 1e-10

    def test_benchmark_vs_itself_gives_nan(self, benchmark_returns, rf_annual):
        """Active returns are all zero, tracking error = 0, IR must be NaN."""
        result = compute_core_metrics(benchmark_returns, benchmark_returns, rf_annual)
        assert np.isnan(result["Information Ratio"])


class TestTrackingError:
    def test_benchmark_vs_itself_is_zero(self, benchmark_returns, rf_annual):
        result = compute_core_metrics(benchmark_returns, benchmark_returns, rf_annual)
        assert abs(result["Tracking Error (%)"]) < 1e-10

    def test_known_value(self, fund_returns, benchmark_returns, rf_annual):
        result = compute_core_metrics(fund_returns, benchmark_returns, rf_annual)
        active = fund_returns - benchmark_returns
        expected = active.std() * np.sqrt(12) * 100
        assert abs(result["Tracking Error (%)"] - expected) < 1e-10
