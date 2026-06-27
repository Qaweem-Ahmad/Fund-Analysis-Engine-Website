"""Shared fixtures for all test modules."""
import numpy as np
import pandas as pd
import pytest


# ── Deterministic synthetic data ─────────────────────────────────────────────

@pytest.fixture
def rf_annual():
    return 0.05

@pytest.fixture
def rf_monthly(rf_annual):
    return rf_annual / 12


@pytest.fixture
def benchmark_returns():
    """60 months of synthetic benchmark log returns (~8% ann.)"""
    rng = np.random.default_rng(42)
    mu = np.log(1.08) / 12
    sigma = 0.04
    return pd.Series(rng.normal(mu, sigma, 60), name="Benchmark")


@pytest.fixture
def fund_returns(benchmark_returns):
    """Fund with positive alpha and beta < 1 vs benchmark."""
    rng = np.random.default_rng(99)
    alpha_monthly = np.log(1.03) / 12   # ~3% annualised alpha
    beta = 0.85
    noise = rng.normal(0, 0.02, len(benchmark_returns))
    return pd.Series(
        alpha_monthly + beta * benchmark_returns.values + noise,
        name="Fund A"
    )


@pytest.fixture
def flat_returns():
    """60 months of exactly zero returns — edge case."""
    return pd.Series(np.zeros(60), name="Flat")


@pytest.fixture
def metrics_matrix_fixture(fund_returns, benchmark_returns):
    """
    metrics_matrix: metrics as ROWS, funds as COLUMNS.
    This is the orientation used by compute_naive_ranking / compute_borda_ranking.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from modules.metrics import compute_core_metrics, compute_downside_metrics

    rf = 0.05
    core   = compute_core_metrics(fund_returns, benchmark_returns, rf)
    down   = compute_downside_metrics(fund_returns, benchmark_returns)
    all_m  = {**core, **down}
    return pd.DataFrame(all_m, index=["Fund A"]).T   # metrics as rows, 1 fund col


@pytest.fixture
def two_fund_metrics_matrix(benchmark_returns):
    """
    Two funds vs the same benchmark — metrics as rows, funds as columns.
    Fund A has higher Sharpe; Fund B has lower Sharpe but lower volatility.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from modules.metrics import compute_core_metrics, compute_downside_metrics

    rng = np.random.default_rng(7)
    rf = 0.05

    fund_a = pd.Series(np.log(1.10) / 12 + 0.85 * benchmark_returns.values + rng.normal(0, 0.02, 60))
    fund_b = pd.Series(np.log(1.06) / 12 + 0.60 * benchmark_returns.values + rng.normal(0, 0.01, 60))

    core_a = compute_core_metrics(fund_a, benchmark_returns, rf)
    down_a = compute_downside_metrics(fund_a, benchmark_returns)
    core_b = compute_core_metrics(fund_b, benchmark_returns, rf)
    down_b = compute_downside_metrics(fund_b, benchmark_returns)

    data = {
        "Fund A": {**core_a, **down_a},
        "Fund B": {**core_b, **down_b},
    }
    return pd.DataFrame(data)   # metrics as rows, funds as columns
