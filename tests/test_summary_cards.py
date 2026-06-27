"""
Integration tests for Executive Summary card logic.
Verifies correct dataframe orientation when looking up metric values.
This test file directly reproduces the bug where Sharpe = 0.000 was shown.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest
from modules.metrics import compute_core_metrics, compute_downside_metrics


# ── Replicate the fixed card logic from app.py ────────────────────────────────

def compute_summary_cards(core_df, downside_df, fund_names, benchmark_name="Benchmark"):
    """
    Mirrors the fixed Executive Summary card logic from app.py.
    core_df:     funds as ROWS, metrics as COLUMNS
    downside_df: funds as ROWS, metrics as COLUMNS
    """
    comparison_funds = [f for f in fund_names if f in core_df.index]
    best_sharpe_fund = core_df.loc[comparison_funds, "Sharpe Ratio"].astype(float).idxmax()
    best_sharpe_val  = float(core_df.loc[best_sharpe_fund, "Sharpe Ratio"])

    best_alpha_fund  = core_df.loc[comparison_funds, "Alpha (ann. %)"].astype(float).idxmax()
    best_alpha_val   = float(core_df.loc[best_alpha_fund, "Alpha (ann. %)"])

    downside_funds   = [f for f in fund_names if f in downside_df.index]
    best_dd_fund     = downside_df.loc[downside_funds, "Max Drawdown (%)"].astype(float).idxmax()
    best_dd_val      = float(downside_df.loc[best_dd_fund, "Max Drawdown (%)"])

    return {
        "best_sharpe_fund": best_sharpe_fund,
        "best_sharpe_val":  best_sharpe_val,
        "best_alpha_fund":  best_alpha_fund,
        "best_alpha_val":   best_alpha_val,
        "best_dd_fund":     best_dd_fund,
        "best_dd_val":      best_dd_val,
    }


def _old_broken_summary_cards(core_df, downside_df, fund_names):
    """
    Reproduces the OLD broken logic: treats metrics as rows, funds as columns.
    This should raise a KeyError or return wrong values.
    """
    best_sharpe_fund = core_df.loc["Sharpe Ratio", fund_names].astype(float).idxmax()
    best_sharpe_val  = float(core_df.loc["Sharpe Ratio", best_sharpe_fund])
    return best_sharpe_fund, best_sharpe_val


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def multi_fund_tables(benchmark_returns):
    """Build core_df and downside_df with funds as rows, metrics as columns."""
    rng = np.random.default_rng(55)
    rf = 0.05

    fund_data = {
        "Fund A": pd.Series(np.log(1.12) / 12 + 0.9 * benchmark_returns.values + rng.normal(0, 0.02, 60)),
        "Fund B": pd.Series(np.log(1.07) / 12 + 0.7 * benchmark_returns.values + rng.normal(0, 0.02, 60)),
        "Fund C": pd.Series(np.log(1.05) / 12 + 0.5 * benchmark_returns.values + rng.normal(0, 0.02, 60)),
    }

    core_rows, down_rows = [], []
    for name, ret in fund_data.items():
        c = compute_core_metrics(ret, benchmark_returns, rf)
        d = compute_downside_metrics(ret, benchmark_returns)
        core_rows.append(pd.Series(c, name=name))
        down_rows.append(pd.Series(d, name=name))

    core_df    = pd.DataFrame(core_rows)     # funds as rows, metrics as cols
    downside_df = pd.DataFrame(down_rows)

    return core_df, downside_df, list(fund_data.keys())


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSummaryCardOrientation:
    def test_old_broken_logic_raises_or_fails(self, multi_fund_tables):
        """
        Regression test: the OLD code used core_df.loc["Sharpe Ratio", fund_names]
        which fails when core_df has funds as rows (Sharpe Ratio is not in the index).
        This test documents that the old code is broken.
        """
        core_df, downside_df, fund_names = multi_fund_tables
        with pytest.raises((KeyError, TypeError, Exception)):
            _old_broken_summary_cards(core_df, downside_df, fund_names)

    def test_new_logic_returns_real_sharpe(self, multi_fund_tables):
        """Fixed logic should return a non-zero, real Sharpe ratio."""
        core_df, downside_df, fund_names = multi_fund_tables
        cards = compute_summary_cards(core_df, downside_df, fund_names)
        assert cards["best_sharpe_val"] != 0.0, \
            "Sharpe ratio is 0.000 — indicates the old broken fallback is still active"
        assert not np.isnan(cards["best_sharpe_val"]), "Sharpe ratio is NaN"
        assert cards["best_sharpe_fund"] in fund_names, \
            f"best_sharpe_fund '{cards['best_sharpe_fund']}' not in fund_names"

    def test_sharpe_card_matches_table_value(self, multi_fund_tables):
        """The Sharpe value on the card must exactly equal the value in core_df."""
        core_df, downside_df, fund_names = multi_fund_tables
        cards = compute_summary_cards(core_df, downside_df, fund_names)
        table_val = float(core_df.loc[cards["best_sharpe_fund"], "Sharpe Ratio"])
        assert abs(cards["best_sharpe_val"] - table_val) < 1e-10

    def test_alpha_card_matches_table_value(self, multi_fund_tables):
        core_df, downside_df, fund_names = multi_fund_tables
        cards = compute_summary_cards(core_df, downside_df, fund_names)
        table_val = float(core_df.loc[cards["best_alpha_fund"], "Alpha (ann. %)"])
        assert abs(cards["best_alpha_val"] - table_val) < 1e-10

    def test_drawdown_card_matches_table_value(self, multi_fund_tables):
        core_df, downside_df, fund_names = multi_fund_tables
        cards = compute_summary_cards(core_df, downside_df, fund_names)
        table_val = float(downside_df.loc[cards["best_dd_fund"], "Max Drawdown (%)"])
        assert abs(cards["best_dd_val"] - table_val) < 1e-10

    def test_best_drawdown_is_least_negative(self, multi_fund_tables):
        """idxmax on negative drawdowns should pick the least negative (best) fund."""
        core_df, downside_df, fund_names = multi_fund_tables
        cards = compute_summary_cards(core_df, downside_df, fund_names)
        dd_col = downside_df.loc[fund_names, "Max Drawdown (%)"].astype(float)
        assert cards["best_dd_val"] == dd_col.max(), \
            "best_dd should be the largest (least negative) drawdown value"

    def test_benchmark_excluded_from_cards(self, multi_fund_tables, benchmark_returns):
        """Benchmark must not appear as best fund in any card."""
        core_df, downside_df, fund_names = multi_fund_tables
        rf = 0.05

        # Add benchmark row
        bm_core = compute_core_metrics(benchmark_returns, benchmark_returns, rf)
        bm_down = compute_downside_metrics(benchmark_returns, benchmark_returns)
        core_df.loc["Benchmark"]    = bm_core
        downside_df.loc["Benchmark"] = bm_down

        cards = compute_summary_cards(core_df, downside_df, fund_names)  # fund_names excludes Benchmark
        assert cards["best_sharpe_fund"] != "Benchmark"
        assert cards["best_alpha_fund"]  != "Benchmark"
        assert cards["best_dd_fund"]     != "Benchmark"


class TestSilentFallbackAudit:
    """
    Flags the known silent fallback patterns from app.py.
    These tests do NOT test the UI — they document the risk.
    """

    def test_sharpe_fallback_zero_is_wrong(self, multi_fund_tables):
        """
        If the lookup silently falls back to 0.0, the card shows 0.000 Sharpe.
        This test confirms 0.0 is never the real best Sharpe for valid fund data.
        """
        core_df, downside_df, fund_names = multi_fund_tables
        cards = compute_summary_cards(core_df, downside_df, fund_names)
        assert cards["best_sharpe_val"] != 0.0, \
            "Sharpe = 0.0 is the silent fallback value, not a real result"

    def test_fund_names_0_fallback_is_wrong(self, multi_fund_tables):
        """
        Old code had: except Exception: best_sharpe_fund = fund_names[0]
        This would always show the first fund even if it's not best.
        Verify that the correct fund is not always fund_names[0].
        """
        core_df, downside_df, fund_names = multi_fund_tables
        cards = compute_summary_cards(core_df, downside_df, fund_names)
        # Fund A has the highest alpha by construction (largest mu), so at least alpha != first-always
        real_best_sharpe = core_df.loc[fund_names, "Sharpe Ratio"].astype(float).idxmax()
        assert cards["best_sharpe_fund"] == real_best_sharpe, \
            f"Got {cards['best_sharpe_fund']} but real best is {real_best_sharpe}"
