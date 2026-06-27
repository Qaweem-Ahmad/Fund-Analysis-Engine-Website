"""Tests for the TOPSIS module (modules/topsis.py)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest
from modules.topsis import TOPSIS


# ── Fixtures ──────────────────────────────────────────────────────────────────

BENEFITS = ["Ann. Return (%)", "Sharpe Ratio"]
COSTS    = ["Ann. Volatility (%)"]

DEFAULT_WEIGHTS = {
    "Returns":  0.30,
    "Risk-Adj": 0.30,
    "Risk/DD":  0.25,
    "Costs":    0.10,
    "ESG":      0.05,
}


@pytest.fixture
def simple_topsis_matrix():
    """
    3-fund, 3-metric matrix. TOPSIS class expects: metrics as ROWS, funds as COLUMNS.
    Fund A dominates on benefits; Fund C dominates on cost.
    """
    data = {
        "Fund A": {"Ann. Return (%)": 12.0, "Sharpe Ratio": 0.9, "Ann. Volatility (%)": 14.0},
        "Fund B": {"Ann. Return (%)":  8.0, "Sharpe Ratio": 0.5, "Ann. Volatility (%)": 18.0},
        "Fund C": {"Ann. Return (%)":  6.0, "Sharpe Ratio": 0.3, "Ann. Volatility (%)": 10.0},
    }
    return pd.DataFrame(data)


@pytest.fixture
def topsis_obj(simple_topsis_matrix):
    return TOPSIS(simple_topsis_matrix, benefits=BENEFITS, costs=COSTS)


@pytest.fixture
def full_metrics_topsis(two_fund_metrics_matrix):
    """Use the full 15-metric matrix for integration-style tests."""
    FULL_BENEFITS = [
        "Ann. Return (%)", "Alpha (ann. %)", "Sharpe Ratio", "Sortino Ratio",
        "Treynor Ratio", "Information Ratio", "R²", "Upside Capture (%)",
        "Calmar Ratio", "Max Drawdown (%)",
    ]
    FULL_COSTS = [
        "Ann. Volatility (%)", "Beta", "Tracking Error (%)",
        "Max DD Duration (mths)", "Downside Capture (%)",
    ]
    return TOPSIS(two_fund_metrics_matrix, benefits=FULL_BENEFITS, costs=FULL_COSTS)


# ── Normalisation ─────────────────────────────────────────────────────────────

class TestNormalisation:
    def test_vector_norm_per_metric_row(self, topsis_obj, simple_topsis_matrix):
        """Each metric row must normalise to unit L2 norm across funds."""
        norm = topsis_obj.normalise()
        for metric in simple_topsis_matrix.index:
            row_norm = np.sqrt((norm.loc[metric] ** 2).sum())
            assert abs(row_norm - 1.0) < 1e-10, \
                f"Row '{metric}' L2 norm = {row_norm}, expected 1.0"

    def test_normalisation_axis_is_row_not_column(self, topsis_obj):
        """Bug regression: normalise() must divide by row L2 norm (axis=1 in old bug)."""
        norm = topsis_obj.normalise()
        # Column norms should NOT be 1 (unless matrix is 1x1)
        col_norms = np.sqrt((norm ** 2).sum(axis=0))
        assert not all(abs(col_norms - 1.0) < 1e-8), \
            "Column norms are all 1 — suggests normalisation is on wrong axis"


# ── Distance calculation ───────────────────────────────────────────────────────

class TestDistances:
    def test_distances_are_non_negative(self, topsis_obj):
        weighted = topsis_obj.weight(DEFAULT_WEIGHTS)
        A_plus, A_minus = topsis_obj.ideal_solutions(weighted)
        D_plus, D_minus = topsis_obj.distances(weighted, A_plus, A_minus)
        assert (D_plus >= 0).all()
        assert (D_minus >= 0).all()

    def test_distance_alignment(self, topsis_obj):
        """D_plus and D_minus must be indexed by fund names (columns), not metrics (rows)."""
        weighted = topsis_obj.weight(DEFAULT_WEIGHTS)
        A_plus, A_minus = topsis_obj.ideal_solutions(weighted)
        D_plus, D_minus = topsis_obj.distances(weighted, A_plus, A_minus)
        for fund in ["Fund A", "Fund B", "Fund C"]:
            assert fund in D_plus.index, f"Fund '{fund}' missing from D_plus index"
            assert fund in D_minus.index, f"Fund '{fund}' missing from D_minus index"


# ── Scores ────────────────────────────────────────────────────────────────────

class TestScores:
    def test_scores_between_0_and_1(self, topsis_obj):
        weighted = topsis_obj.weight(DEFAULT_WEIGHTS)
        A_plus, A_minus = topsis_obj.ideal_solutions(weighted)
        D_plus, D_minus = topsis_obj.distances(weighted, A_plus, A_minus)
        scores = topsis_obj.score(D_plus, D_minus)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_dominant_fund_scores_highest(self, topsis_obj):
        """Fund A has the best return and Sharpe — should win."""
        weighted = topsis_obj.weight(DEFAULT_WEIGHTS)
        A_plus, A_minus = topsis_obj.ideal_solutions(weighted)
        D_plus, D_minus = topsis_obj.distances(weighted, A_plus, A_minus)
        scores = topsis_obj.score(D_plus, D_minus)
        assert scores.idxmax() == "Fund A", \
            f"Expected Fund A to score highest, got {scores.idxmax()}"

    def test_identical_funds_score_the_same(self):
        """Two identical funds should receive identical TOPSIS scores."""
        data = {
            "Fund X": {"Ann. Return (%)": 10.0, "Sharpe Ratio": 0.7, "Ann. Volatility (%)": 15.0},
            "Fund Y": {"Ann. Return (%)": 10.0, "Sharpe Ratio": 0.7, "Ann. Volatility (%)": 15.0},
        }
        t = TOPSIS(pd.DataFrame(data), benefits=BENEFITS, costs=COSTS)
        weighted = t.weight(DEFAULT_WEIGHTS)
        A_plus, A_minus = t.ideal_solutions(weighted)
        D_plus, D_minus = t.distances(weighted, A_plus, A_minus)
        scores = t.score(D_plus, D_minus)
        assert abs(scores["Fund X"] - scores["Fund Y"]) < 1e-10


# ── Full run ──────────────────────────────────────────────────────────────────

class TestFullRun:
    def test_run_returns_dataframe(self, full_metrics_topsis):
        result = full_metrics_topsis.run(DEFAULT_WEIGHTS)
        assert isinstance(result, pd.DataFrame)

    def test_run_contains_rank_and_score(self, full_metrics_topsis):
        result = full_metrics_topsis.run(DEFAULT_WEIGHTS)
        assert "Rank" in result.columns
        assert "Score" in result.columns

    def test_ranks_start_at_1(self, full_metrics_topsis):
        result = full_metrics_topsis.run(DEFAULT_WEIGHTS)
        assert result["Rank"].min() == 1

    def test_ranks_are_sequential(self, full_metrics_topsis):
        result = full_metrics_topsis.run(DEFAULT_WEIGHTS)
        n = len(result)
        assert sorted(result["Rank"].tolist()) == list(range(1, n + 1))
