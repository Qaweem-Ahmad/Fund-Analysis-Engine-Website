"""
Regression tests for modules/charts.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from modules.charts import chart_cmatrix


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
