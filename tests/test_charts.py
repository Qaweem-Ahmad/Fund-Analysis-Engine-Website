"""
Regression tests for modules/charts.py and app.py chart functions.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from modules.charts import chart_cmatrix


APP_PATH = os.path.join(os.path.dirname(__file__), '..', 'app.py')


def _corr_heatmap_body():
    """Return source text of chart_correlation_heatmap function body."""
    with open(APP_PATH) as f:
        src = f.read()
    start = src.find('def chart_correlation_heatmap(')
    end = src.find('\ndef ', start + 1)
    return src[start:end]


FUNDS = ["Fund A", "Fund B", "Fund C"]


@pytest.fixture
def int_cmatrix():
    """Integer-dtype C-matrix -- the dtype that triggered the ValueError on Streamlit Cloud."""
    data = [
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
    ]
    return pd.DataFrame(data, index=FUNDS, columns=FUNDS)


@pytest.fixture
def float_cmatrix():
    data = [
        [1.0, 0.3, 0.7],
        [0.7, 1.0, 0.4],
        [0.3, 0.6, 1.0],
    ]
    return pd.DataFrame(data, index=FUNDS, columns=FUNDS)


class TestChartCmatrix:
    def test_int_dtype_does_not_raise(self, int_cmatrix):
        """Regression: np.fill_diagonal(np.nan) on int dtype used to raise ValueError."""
        chart_cmatrix(int_cmatrix, {})  # must not raise

    def test_float_dtype_does_not_raise(self, float_cmatrix):
        chart_cmatrix(float_cmatrix, {})  # must not raise

    def test_diagonal_is_nan_in_output(self, float_cmatrix):
        """Diagonal cells must be NaN so Plotly renders them as blank."""
        import plotly.graph_objects as go
        fig = chart_cmatrix(float_cmatrix, {})
        z = fig.data[0].z
        for i in range(len(FUNDS)):
            assert np.isnan(z[i][i]), f"Diagonal [{i}][{i}] should be NaN, got {z[i][i]}"

    def test_original_matrix_not_mutated(self, int_cmatrix):
        """chart_cmatrix must not modify the caller's DataFrame."""
        original_dtype = int_cmatrix.dtypes.iloc[0]
        chart_cmatrix(int_cmatrix, {})
        assert int_cmatrix.dtypes.iloc[0] == original_dtype
        assert int_cmatrix.iloc[0, 0] == 1  # diagonal still 1, not NaN


class TestCorrelationHeatmap:
    """Regression tests for chart_correlation_heatmap in app.py."""

    def test_no_titlefont_in_app(self):
        """app.py must contain zero 'titlefont' occurrences after the Plotly fix."""
        with open(APP_PATH) as f:
            source = f.read()
        assert 'titlefont' not in source, "Found deprecated 'titlefont' in app.py"

    def test_tight_colour_scale(self):
        """Correlation heatmap must use zmin=0.80/zmax=1.00, not -1/+1."""
        body = _corr_heatmap_body()
        assert 'zmin=0.80' in body, "Expected tight zmin=0.80 in chart_correlation_heatmap"
        assert 'zmax=1.00' in body, "Expected tight zmax=1.00 in chart_correlation_heatmap"

    def test_three_dp_labels(self):
        """Correlation text labels must be formatted to 3 decimal places."""
        body = _corr_heatmap_body()
        assert '.3f' in body, "Expected ':.3f' format in chart_correlation_heatmap text labels"

    def test_diagonal_muted(self):
        """Diagonal z-values must be set to NaN so off-diagonal colours fill the scale."""
        body = _corr_heatmap_body()
        assert 'fill_diagonal' in body, "Expected np.fill_diagonal to mute diagonal in correlation heatmap"
        assert 'nan' in body.lower(), "Expected NaN assignment for diagonal in correlation heatmap"
