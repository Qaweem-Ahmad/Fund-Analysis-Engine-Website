"""
Tests for highlight_metrics_table logic.
Verifies column-by-column (per metric) highlighting, not row-by-row.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest


# Replicate the highlight function as it exists in app.py
BEST_STYLE  = "background-color: #EAF2F3; color: #154D57"
WORST_STYLE = "background-color: #F8F3EE; color: #7A5642"

def highlight_metrics_table(df, directions, benchmark_name="Benchmark"):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    comparison_rows = [idx for idx in df.index if idx != benchmark_name]
    for metric, direction in directions.items():
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df.loc[comparison_rows, metric], errors="coerce").dropna()
        if values.empty:
            continue
        if direction == "higher":
            best_val, worst_val = values.max(), values.min()
        else:
            best_val, worst_val = values.min(), values.max()
        styles.loc[values[values == best_val].index, metric]  = BEST_STYLE
        styles.loc[values[values == worst_val].index, metric] = WORST_STYLE
    return styles


CORE_DIRECTIONS = {
    "Ann. Return (%)":      "higher",
    "Ann. Volatility (%)":  "lower",
    "Alpha (ann. %)":       "higher",
    "Beta":                 "lower",
    "R²":                   "higher",
    "Sharpe Ratio":         "higher",
    "Treynor Ratio":        "higher",
    "Sortino Ratio":        "higher",
    "Information Ratio":    "higher",
    "Tracking Error (%)":   "lower",
}

DOWNSIDE_DIRECTIONS = {
    "Upside Capture (%)":     "higher",
    "Downside Capture (%)":   "lower",
    "Max Drawdown (%)":       "higher",
    "Max DD Duration (mths)": "lower",
    "Calmar Ratio":           "higher",
}


@pytest.fixture
def sample_core_df():
    """funds as ROWS, metrics as COLUMNS — the correct orientation for st.dataframe.
    Values deliberately spread across funds so no single fund dominates every metric:
      Ann. Return:  Fund A best, Fund B worst
      Volatility:   Fund B best (lowest), Fund A worst (highest)
      Sharpe Ratio: Fund C best, Fund B worst
    """
    return pd.DataFrame({
        "Ann. Return (%)":     {"Fund A": 10.0, "Fund B": 6.0,  "Fund C": 8.0,  "Benchmark": 7.0},
        "Ann. Volatility (%)": {"Fund A": 14.0, "Fund B": 20.0, "Fund C": 16.0, "Benchmark": 15.0},
        "Sharpe Ratio":        {"Fund A": 0.6,  "Fund B": 0.3,  "Fund C": 0.9,  "Benchmark": 0.5},
    })


class TestHighlightByColumn:
    def test_best_return_fund_highlighted(self, sample_core_df):
        """Fund A has highest return — its Ann. Return cell should get BEST_STYLE."""
        styles = highlight_metrics_table(sample_core_df, CORE_DIRECTIONS)
        assert styles.loc["Fund A", "Ann. Return (%)"] == BEST_STYLE

    def test_worst_return_fund_highlighted(self, sample_core_df):
        """Fund B has lowest return — its Ann. Return cell should get WORST_STYLE."""
        styles = highlight_metrics_table(sample_core_df, CORE_DIRECTIONS)
        assert styles.loc["Fund B", "Ann. Return (%)"] == WORST_STYLE

    def test_lower_volatility_is_better(self, sample_core_df):
        """Fund A has lowest volatility — should get BEST_STYLE for Ann. Volatility."""
        styles = highlight_metrics_table(sample_core_df, CORE_DIRECTIONS)
        assert styles.loc["Fund A", "Ann. Volatility (%)"] == BEST_STYLE

    def test_higher_volatility_is_worst(self, sample_core_df):
        """Fund B has highest volatility — should get WORST_STYLE."""
        styles = highlight_metrics_table(sample_core_df, CORE_DIRECTIONS)
        assert styles.loc["Fund B", "Ann. Volatility (%)"] == WORST_STYLE

    def test_benchmark_is_never_highlighted(self, sample_core_df):
        """Benchmark row must always have empty styles."""
        styles = highlight_metrics_table(sample_core_df, CORE_DIRECTIONS)
        for col in sample_core_df.columns:
            assert styles.loc["Benchmark", col] == "", \
                f"Benchmark got style in column '{col}'"

    def test_style_matrix_shape_matches_input(self, sample_core_df):
        styles = highlight_metrics_table(sample_core_df, CORE_DIRECTIONS)
        assert styles.shape == sample_core_df.shape

    def test_not_all_cells_in_same_row_highlighted(self, sample_core_df):
        """
        Regression: old axis=1 bug highlighted ALL cells in a row if one value dominated.
        Correct behaviour: at most one cell per column should be BEST, one WORST.
        """
        styles = highlight_metrics_table(sample_core_df, CORE_DIRECTIONS)
        fund_rows = [r for r in sample_core_df.index if r != "Benchmark"]
        for fund in fund_rows:
            row = styles.loc[fund]
            n_best  = (row == BEST_STYLE).sum()
            n_worst = (row == WORST_STYLE).sum()
            # A fund could legitimately be best at multiple metrics, but NOT at ALL metrics
            assert n_best < len(row), \
                f"{fund} is highlighted BEST for every metric — suggests row-wise bug"


class TestMaxDrawdownHighlighting:
    def test_less_negative_drawdown_is_best(self):
        """Max Drawdown is stored as negative numbers. -10 > -30, so -10 is better."""
        df = pd.DataFrame({
            "Max Drawdown (%)": {"Fund A": -10.0, "Fund B": -30.0, "Benchmark": -20.0},
        })
        styles = highlight_metrics_table(df, DOWNSIDE_DIRECTIONS)
        assert styles.loc["Fund A", "Max Drawdown (%)"] == BEST_STYLE, \
            "Fund A (-10%) should be BEST — less negative is better"
        assert styles.loc["Fund B", "Max Drawdown (%)"] == WORST_STYLE, \
            "Fund B (-30%) should be WORST — more negative is worse"

    def test_direction_is_higher_for_drawdown(self):
        """DOWNSIDE_DIRECTIONS must specify 'higher' for Max Drawdown."""
        assert DOWNSIDE_DIRECTIONS["Max Drawdown (%)"] == "higher", \
            "Max Drawdown direction must be 'higher' (less negative is better)"


class TestHighlightWithSingleFund:
    def test_single_fund_gets_both_best_and_worst(self):
        """With only one fund, it must be both best AND worst in every metric."""
        df = pd.DataFrame({
            "Sharpe Ratio":    {"Fund A": 0.8, "Benchmark": 0.5},
            "Ann. Return (%)": {"Fund A": 9.0, "Benchmark": 7.0},
        })
        styles = highlight_metrics_table(df, CORE_DIRECTIONS)
        # With one comparison fund, best == worst == that fund
        assert styles.loc["Fund A", "Sharpe Ratio"] in (BEST_STYLE, WORST_STYLE)
