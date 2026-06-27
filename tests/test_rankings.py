"""Tests for naive ranking, Borda count ranking, and dataframe orientation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest


# Import ranking functions directly from app module scope
# We extract them by importing app — but app has Streamlit side-effects,
# so we import just the two functions via exec from the source file.
import importlib.util, types

def _load_ranking_funcs():
    src_path = os.path.join(os.path.dirname(__file__), '..', 'app.py')
    with open(src_path) as f:
        source = f.read()
    # Extract only the function definitions we need
    import ast
    tree = ast.parse(source)
    funcs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            'compute_naive_ranking', 'compute_borda_ranking'
        ):
            funcs[node.name] = node
    mod = types.ModuleType('_ranking_funcs')
    mod.__dict__.update({'pd': pd, 'np': np})
    for name, node in funcs.items():
        code = compile(ast.Module(body=[node], type_ignores=[]), '<string>', 'exec')
        exec(code, mod.__dict__)
    return mod

_rf = _load_ranking_funcs()
compute_naive_ranking  = _rf.compute_naive_ranking
compute_borda_ranking  = _rf.compute_borda_ranking


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_matrix():
    """
    metrics_matrix: metrics as ROWS, funds as COLUMNS (no Benchmark column).
    Fund A dominates on all three naive metrics.
    """
    data = {
        "Fund A": {"Ann. Return (%)": 10.0, "Sharpe Ratio": 0.8, "Max Drawdown (%)": -15.0},
        "Fund B": {"Ann. Return (%)":  6.0, "Sharpe Ratio": 0.4, "Max Drawdown (%)": -30.0},
        "Fund C": {"Ann. Return (%)":  8.0, "Sharpe Ratio": 0.6, "Max Drawdown (%)": -22.0},
    }
    return pd.DataFrame(data)


@pytest.fixture
def uniform_weights():
    return {
        "Ann. Return (%)": 1/19, "Alpha (ann. %)": 1/19,
        "Sharpe Ratio": 1/19, "Sortino Ratio": 1/19,
        "Treynor Ratio": 1/19, "Information Ratio": 1/19, "R²": 1/19,
        "Upside Capture (%)": 1/19, "Calmar Ratio": 1/19,
        "ESG Globe Rating": 1/19, "Ann. Volatility (%)": 1/19,
        "Beta": 1/19, "Tracking Error (%)": 1/19, "Max Drawdown (%)": 1/19,
        "Max DD Duration (mths)": 1/19, "Downside Capture (%)": 1/19,
        "OCF": 1/19, "Carbon Risk Score": 1/19,
    }


# ── DataFrame orientation tests ───────────────────────────────────────────────

class TestDataframeOrientation:
    def test_metrics_matrix_has_metrics_as_rows(self, two_fund_metrics_matrix):
        """metrics_matrix must have metric names as the index."""
        mm = two_fund_metrics_matrix
        assert "Sharpe Ratio" in mm.index, "metrics_matrix should have metrics as rows"
        assert "Fund A" in mm.columns, "metrics_matrix should have funds as columns"

    def test_core_df_has_funds_as_rows(self, fund_returns, benchmark_returns):
        """core_metrics_df (from session state) must have funds as rows."""
        from modules.metrics import compute_core_metrics
        result = compute_core_metrics(fund_returns, benchmark_returns, 0.05)
        df = pd.DataFrame([result], index=["Fund A"])
        assert "Fund A" in df.index
        assert "Sharpe Ratio" in df.columns

    def test_naive_ranking_expects_metrics_as_rows(self, simple_matrix):
        """compute_naive_ranking uses metrics_matrix.loc[metric] — metrics must be rows."""
        result = compute_naive_ranking(simple_matrix)
        assert isinstance(result, pd.DataFrame)
        assert "Naive Rank" in result.columns

    def test_naive_ranking_with_wrong_orientation_fails_or_empty(self, simple_matrix):
        """If matrix is transposed (funds as rows), naive ranking should return empty or wrong result."""
        wrong_orientation = simple_matrix.T   # funds as rows — wrong orientation
        result = compute_naive_ranking(wrong_orientation)
        # The function filters avail metrics from index — if index is fund names, avail will be empty
        assert result.empty or ("Naive Rank" not in result.columns or len(result) != 3), \
            "Naive ranking should break or return empty with wrong orientation"


# ── Naive ranking tests ───────────────────────────────────────────────────────

class TestNaiveRanking:
    def test_dominant_fund_ranks_first(self, simple_matrix):
        result = compute_naive_ranking(simple_matrix)
        assert result.index[0] == "Fund A", f"Expected Fund A first, got {result.index[0]}"

    def test_returns_correct_columns(self, simple_matrix):
        result = compute_naive_ranking(simple_matrix)
        assert "Naive Score (%)" in result.columns
        assert "Naive Rank" in result.columns

    def test_rank_1_has_highest_score(self, simple_matrix):
        result = compute_naive_ranking(simple_matrix)
        rank1_score = result.loc[result["Naive Rank"] == 1, "Naive Score (%)"].iloc[0]
        assert rank1_score == result["Naive Score (%)"].max()

    def test_scores_between_0_and_100(self, simple_matrix):
        result = compute_naive_ranking(simple_matrix)
        assert result["Naive Score (%)"].between(0, 100).all()

    def test_empty_matrix_returns_empty(self):
        result = compute_naive_ranking(pd.DataFrame())
        assert result.empty

    def test_max_drawdown_treated_as_cost(self):
        """Less negative max drawdown should score better, not worse."""
        data = {
            "Good DD": {"Ann. Return (%)": 8.0, "Sharpe Ratio": 0.6, "Max Drawdown (%)": -10.0},
            "Bad DD":  {"Ann. Return (%)": 8.0, "Sharpe Ratio": 0.6, "Max Drawdown (%)": -40.0},
        }
        mm = pd.DataFrame(data)
        result = compute_naive_ranking(mm)
        assert result.index[0] == "Good DD", \
            "Fund with smaller drawdown should rank first when returns/Sharpe are equal"

    def test_excludes_benchmark_column(self):
        data = {
            "Fund A":    {"Ann. Return (%)": 10.0, "Sharpe Ratio": 0.8, "Max Drawdown (%)": -15.0},
            "Fund B":    {"Ann. Return (%)":  6.0, "Sharpe Ratio": 0.4, "Max Drawdown (%)": -30.0},
            "Benchmark": {"Ann. Return (%)":  7.0, "Sharpe Ratio": 0.5, "Max Drawdown (%)": -20.0},
        }
        result = compute_naive_ranking(pd.DataFrame(data))
        assert "Benchmark" not in result.index


# ── Borda ranking tests ───────────────────────────────────────────────────────

class TestBordaRanking:
    def test_dominant_fund_ranks_first(self, simple_matrix, uniform_weights):
        result = compute_borda_ranking(simple_matrix, uniform_weights)
        assert result.index[0] == "Fund A"

    def test_returns_correct_columns(self, simple_matrix, uniform_weights):
        result = compute_borda_ranking(simple_matrix, uniform_weights)
        assert "Borda Score (%)" in result.columns
        assert "Borda Rank" in result.columns

    def test_top_score_is_100(self, simple_matrix, uniform_weights):
        result = compute_borda_ranking(simple_matrix, uniform_weights)
        assert result["Borda Score (%)"].max() == 100.0

    def test_scores_between_0_and_100(self, simple_matrix, uniform_weights):
        result = compute_borda_ranking(simple_matrix, uniform_weights)
        assert result["Borda Score (%)"].between(0, 100).all()

    def test_excludes_benchmark_column(self, uniform_weights):
        data = {
            "Fund A":    {"Ann. Return (%)": 10.0, "Sharpe Ratio": 0.8, "Max Drawdown (%)": -15.0},
            "Fund B":    {"Ann. Return (%)":  6.0, "Sharpe Ratio": 0.4, "Max Drawdown (%)": -30.0},
            "Benchmark": {"Ann. Return (%)":  7.0, "Sharpe Ratio": 0.5, "Max Drawdown (%)": -20.0},
        }
        result = compute_borda_ranking(pd.DataFrame(data), uniform_weights)
        assert "Benchmark" not in result.index

    def test_missing_metric_in_weights_is_skipped(self, uniform_weights):
        data = {"Fund A": {"Ann. Return (%)": 10.0}, "Fund B": {"Ann. Return (%)": 6.0}}
        # weights reference many metrics not in this tiny matrix - should not raise
        result = compute_borda_ranking(pd.DataFrame(data), uniform_weights)
        assert not result.empty
