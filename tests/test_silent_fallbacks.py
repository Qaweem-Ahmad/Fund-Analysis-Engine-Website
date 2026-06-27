"""
Audit of silent fallback patterns in app.py.
These tests flag high-risk except/fallback patterns — they do not fix them.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import re
import pytest

APP_PATH = os.path.join(os.path.dirname(__file__), '..', 'app.py')


def _read_app():
    with open(APP_PATH) as f:
        return f.read()

def _find_pattern(source, pattern, flags=0):
    return [(m.start(), source[:m.start()].count('\n') + 1, m.group())
            for m in re.finditer(pattern, source, flags)]


class TestSilentFallbacks:
    """
    Static analysis of app.py for dangerous silent fallback patterns.
    Each test flags a risk category and reports the line numbers found.
    Failing tests mean the pattern EXISTS — not that it has been fixed.
    """

    def test_no_bare_except_pass(self):
        src = _read_app()
        hits = _find_pattern(src, r'except\s+Exception\s*:\s*\n\s*pass')
        assert not hits, (
            f"Found {len(hits)} bare 'except Exception: pass' blocks "
            f"at lines: {[ln for _, ln, _ in hits]}\n"
            "These silently swallow errors. Replace with st.error() or logging."
        )

    def test_no_bare_except_continue(self):
        src = _read_app()
        hits = _find_pattern(src, r'except\s+Exception\s*:\s*\n\s*continue')
        assert not hits, (
            f"Found {len(hits)} 'except Exception: continue' blocks "
            f"at lines: {[ln for _, ln, _ in hits]}\n"
            "These silently skip failures in loops."
        )

    def test_no_fund_names_0_fallback(self):
        """
        The old executive summary bug: except Exception: best_X = fund_names[0]
        This always shows the first fund regardless of actual ranking.
        """
        src = _read_app()
        hits = _find_pattern(src, r'fund_names\[0\]')
        assert not hits, (
            f"Found {len(hits)} 'fund_names[0]' fallback(s) "
            f"at lines: {[ln for _, ln, _ in hits]}\n"
            "HIGH RISK: these default to the first fund silently on exception."
        )

    def test_no_zero_point_zero_fallback_in_except(self):
        """
        Old pattern: except Exception: best_val = 0.0
        Shows 0.000 on summary cards when the real lookup fails.
        """
        src = _read_app()
        # Look for 0.0 assignment inside except blocks
        hits = _find_pattern(src, r'except Exception.*?=\s*0\.0', re.DOTALL)
        # Simple check: find except blocks with 0.0 on next few lines
        blocks = re.findall(
            r'except Exception[^:]*:\s*\n(?:\s*\w+\s*=\s*\w+\[0\]\s*\n)*\s*\w+\s*=\s*0\.0',
            src
        )
        assert not blocks, (
            f"Found {len(blocks)} except block(s) with 0.0 fallback values.\n"
            "These produce incorrect zeros on summary cards."
        )

    def test_no_na_string_fallback(self):
        """
        New pattern after fix: except Exception as e: best_X = 'N/A'
        N/A in cards is better than wrong values, but still flags an issue.
        """
        src = _read_app()
        hits = _find_pattern(src, r'"N/A"')
        fallback_hits = [(pos, ln, txt) for pos, ln, txt in hits
                         if 'best_' in src[max(0,pos-200):pos+50]]
        if fallback_hits:
            pytest.xfail(
                f"Found {len(fallback_hits)} 'N/A' fallback(s) near card logic "
                f"at lines: {[ln for _, ln, _ in fallback_hits]}\n"
                "These are acceptable as error states but should be investigated."
            )

    def test_silent_except_count_in_app(self):
        """
        Count total silent except blocks (no st.error / logging inside them).
        Report as informational — does not fail but documents the risk surface.
        """
        src = _read_app()
        all_excepts = _find_pattern(src, r'except Exception')
        # Check which ones have st.error nearby
        risky = []
        for pos, ln, _ in all_excepts:
            block = src[pos:pos+300]
            if 'st.error' not in block and 'logging' not in block and 'print' not in block:
                risky.append(ln)
        if risky:
            pytest.xfail(
                f"{len(risky)} except block(s) with no visible error reporting "
                f"at lines: {risky}\n"
                "These silently swallow failures. Consider adding st.error()."
            )


class TestOrientationAssumptions:
    """
    Flag code in app.py that assumes a specific dataframe orientation.
    These are informational — they document orientation-sensitive code paths.
    """

    def test_metrics_matrix_row_access_pattern(self):
        """
        compute_naive_ranking uses metrics_matrix.loc[metric, fund_cols].
        This assumes metrics as rows. Verify this pattern exists in the source.
        """
        src = _read_app()
        # The function should access by metric name (a string) as row
        assert 'metrics_matrix.loc[' in src, \
            "metrics_matrix.loc[ not found — naive/borda ranking may have changed"

    def test_core_df_column_access_pattern(self):
        """
        Fixed summary card code accesses core_df.loc[fund, metric].
        Verify the correct (fixed) orientation pattern is present.
        """
        src = _read_app()
        assert 'core_df.loc[comparison_funds' in src or 'core_df.loc[downside_funds' in src or \
               'core_df.loc[comparison_funds,' in src, \
            "Fixed orientation pattern not found in summary card code — may have regressed"

    def test_old_broken_orientation_not_present(self):
        """
        The old broken pattern: core_df.loc["Sharpe Ratio", fund_names]
        (metrics as row label, funds as column label).
        This must NOT be present in the current code.
        """
        src = _read_app()
        broken = re.findall(r'core_df\.loc\["Sharpe Ratio"', src)
        assert not broken, (
            f"Found {len(broken)} instance(s) of old broken orientation: "
            "core_df.loc[\"Sharpe Ratio\", ...]\n"
            "This assumes metrics as rows but core_df has funds as rows."
        )
