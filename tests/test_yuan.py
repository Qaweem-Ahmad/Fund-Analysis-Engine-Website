"""Tests for the Yuan & Yuan eigenvector ranking module (modules/yuan.py)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest
from modules.yuan import YuanYuan


BENEFITS = [
    "Ann. Return (%)", "Alpha (ann. %)", "Sharpe Ratio", "Sortino Ratio",
    "Treynor Ratio", "Information Ratio", "R²", "Upside Capture (%)",
    "Calmar Ratio", "Max Drawdown (%)",
]
COSTS = [
    "Ann. Volatility (%)", "Beta", "Tracking Error (%)",
    "Max DD Duration (mths)", "Downside Capture (%)",
]

DEFAULT_WEIGHTS = {
    "Returns":  0.30,
    "Risk-Adj": 0.30,
    "Risk/DD":  0.25,
    "Costs":    0.10,
    "ESG":      0.05,
}


@pytest.fixture
def yuan_obj(two_fund_metrics_matrix):
    return YuanYuan(two_fund_metrics_matrix, benefits=BENEFITS, costs=COSTS)


class TestCompetitionMatrix:
    def test_shape_is_square(self, yuan_obj):
        cm = yuan_obj.competition_matrix(DEFAULT_WEIGHTS)
        assert cm.shape[0] == cm.shape[1]

    def test_diagonal_is_half(self, yuan_obj):
        """A fund vs itself: for each metric both sides are equal, so score = 0.5*w per metric."""
        cm = yuan_obj.competition_matrix(DEFAULT_WEIGHTS)
        for fund in cm.index:
            assert abs(cm.loc[fund, fund] - 0.5) < 1e-8, \
                f"Diagonal for {fund} = {cm.loc[fund, fund]}, expected 0.5"

    def test_row_pairs_sum_to_1(self, yuan_obj):
        """c(i,j) + c(j,i) must equal 1.0 for any pair."""
        cm = yuan_obj.competition_matrix(DEFAULT_WEIGHTS)
        funds = cm.index.tolist()
        for i in range(len(funds)):
            for j in range(i + 1, len(funds)):
                fi, fj = funds[i], funds[j]
                pair_sum = cm.loc[fi, fj] + cm.loc[fj, fi]
                assert abs(pair_sum - 1.0) < 1e-8, \
                    f"c({fi},{fj}) + c({fj},{fi}) = {pair_sum}, expected 1.0"


class TestPowerIteration:
    def test_converges(self, yuan_obj):
        cm = yuan_obj.competition_matrix(DEFAULT_WEIGHTS)
        scores, iters = yuan_obj.power_iteration(cm)
        assert iters < 1000, f"Power iteration did not converge in 1000 steps"

    def test_scores_are_non_negative(self, yuan_obj):
        cm = yuan_obj.competition_matrix(DEFAULT_WEIGHTS)
        scores, _ = yuan_obj.power_iteration(cm)
        assert (scores >= 0).all()


class TestFullRun:
    def test_scores_sum_to_1(self, yuan_obj):
        """After L1 normalisation, scores must sum to 1."""
        ranking, _, _ = yuan_obj.run(DEFAULT_WEIGHTS)
        assert abs(ranking["Score"].sum() - 1.0) < 1e-8

    def test_rank_column_present(self, yuan_obj):
        ranking, _, _ = yuan_obj.run(DEFAULT_WEIGHTS)
        assert "Rank" in ranking.columns

    def test_ranks_start_at_1(self, yuan_obj):
        ranking, _, _ = yuan_obj.run(DEFAULT_WEIGHTS)
        assert ranking["Rank"].min() == 1

    def test_ranks_are_sequential(self, yuan_obj):
        ranking, _, _ = yuan_obj.run(DEFAULT_WEIGHTS)
        n = len(ranking)
        assert sorted(ranking["Rank"].tolist()) == list(range(1, n + 1))

    def test_identical_funds_rank_equally(self):
        """Two identical funds should have the same score after power iteration."""
        data = {
            "Fund X": {"Ann. Return (%)": 10.0, "Sharpe Ratio": 0.7,
                       "Ann. Volatility (%)": 15.0, "Beta": 0.9,
                       "Max Drawdown (%)": -20.0, "Upside Capture (%)": 100.0,
                       "Downside Capture (%)": 95.0, "Calmar Ratio": 0.5,
                       "Alpha (ann. %)": 1.0, "Sortino Ratio": 0.6,
                       "Treynor Ratio": 0.05, "Information Ratio": 0.3,
                       "R²": 0.85, "Tracking Error (%)": 4.0,
                       "Max DD Duration (mths)": 12.0},
            "Fund Y": {"Ann. Return (%)": 10.0, "Sharpe Ratio": 0.7,
                       "Ann. Volatility (%)": 15.0, "Beta": 0.9,
                       "Max Drawdown (%)": -20.0, "Upside Capture (%)": 100.0,
                       "Downside Capture (%)": 95.0, "Calmar Ratio": 0.5,
                       "Alpha (ann. %)": 1.0, "Sortino Ratio": 0.6,
                       "Treynor Ratio": 0.05, "Information Ratio": 0.3,
                       "R²": 0.85, "Tracking Error (%)": 4.0,
                       "Max DD Duration (mths)": 12.0},
        }
        df = pd.DataFrame(data)
        yy = YuanYuan(df, benefits=BENEFITS, costs=COSTS)
        ranking, _, _ = yy.run(DEFAULT_WEIGHTS)
        scores = ranking["Score"]
        assert abs(scores.iloc[0] - scores.iloc[1]) < 1e-8, \
            f"Identical funds have different scores: {scores.tolist()}"
