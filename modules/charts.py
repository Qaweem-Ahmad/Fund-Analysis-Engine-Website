import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict


def chart_cumulative_returns(log_returns: pd.DataFrame, colours: Dict[str, str], plotly_layout: Dict) -> go.Figure:
    fig = go.Figure()
    for col in log_returns.columns:
        cum_ret = (np.exp(log_returns[col].cumsum()) - 1) * 100
        if col == 'Benchmark':
            fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, mode='lines', name=col, line=dict(dash='dot', color='#6E6E73')))
        else:
            fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, mode='lines', name=col, line=dict(color=colours.get(col, '#0071E3'))))
    fig.add_hline(y=0, line_dash="dash", line_color="#E5E5EA")
    fig.update_layout(**{k: v for k, v in plotly_layout.items() if k != 'title'})
    fig.update_layout(title=dict(text="Cumulative Returns: January 2020 to October 2025", font=dict(size=16, color="#1D1D1F")))
    fig.update_layout(
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", yanchor="top"),
        margin=dict(b=55),
    )
    return fig


def chart_drawdown(fund_returns: pd.DataFrame, benchmark_returns: pd.Series, colours: Dict[str, str], plotly_layout: Dict) -> go.Figure:
    fig = go.Figure()
    for col in fund_returns.columns:
        wealth = np.exp(fund_returns[col].cumsum())
        cummax = wealth.cummax()
        drawdown = (wealth - cummax) / cummax * 100
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', mode='lines', name=col, line=dict(color=colours.get(col, '#0071E3'), width=1)))
    # Benchmark
    wealth_b = np.exp(benchmark_returns.cumsum())
    cummax_b = wealth_b.cummax()
    drawdown_b = (wealth_b - cummax_b) / cummax_b * 100
    fig.add_trace(go.Scatter(x=drawdown_b.index, y=drawdown_b, mode='lines', name='Benchmark', line=dict(dash='dot', color='#6E6E73')))
    fig.update_layout(**{k: v for k, v in plotly_layout.items() if k != 'title'})
    fig.update_layout(title=dict(text="Peak-to-Trough Drawdown", font=dict(size=16, color="#1D1D1F")))
    fig.update_layout(
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", yanchor="top"),
        margin=dict(b=55),
    )
    return fig


def chart_rolling_sharpe(fund_returns: pd.DataFrame, rf_monthly: float, colours: Dict[str, str], plotly_layout: Dict) -> go.Figure:
    fig = go.Figure()
    for col in fund_returns.columns:
        excess = fund_returns[col] - rf_monthly
        rolling_mean = excess.rolling(12).mean() * 12
        rolling_std = excess.rolling(12).std() * np.sqrt(12)
        rolling_sharpe = rolling_mean / rolling_std
        fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, mode='lines', name=col, line=dict(color=colours.get(col, '#0071E3'))))
    fig.add_hline(y=0, line_dash="dash", line_color="#E5E5EA")
    fig.update_layout(**{k: v for k, v in plotly_layout.items() if k != 'title'})
    fig.update_layout(title=dict(text="Rolling 12-Month Sharpe Ratio", font=dict(size=16, color="#1D1D1F")))
    fig.update_layout(
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", yanchor="top"),
        margin=dict(b=55),
    )
    return fig


def chart_topsis_heatmap(normalised_df: pd.DataFrame, plotly_layout: Dict) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=normalised_df.values,
        x=normalised_df.columns,
        y=normalised_df.index,
        colorscale=[[0, '#FF3B30'], [0.5, '#FF9F0A'], [1, '#34C759']],
        text=np.round(normalised_df.values, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    fig.update_layout(**{k: v for k, v in plotly_layout.items() if k != 'title'})
    fig.update_layout(title=dict(text="TOPSIS Input Matrix — Min-Max Normalised Scores", font=dict(size=16, color="#1D1D1F")))
    return fig


def chart_cmatrix(c_matrix: pd.DataFrame, plotly_layout: Dict) -> go.Figure:
    labels = list(c_matrix.columns)
    n = len(labels)

    # Writable NumPy copy -- to_numpy(copy=True) avoids read-only buffer issues
    c_vals = c_matrix.to_numpy(dtype=float, copy=True)

    # Trace 1: off-diagonal competition scores (diagonal = NaN)
    main_z = c_vals.copy()
    np.fill_diagonal(main_z, np.nan)

    # Trace 2: diagonal overlay -- neutral colour, plain "-" to indicate self-comparison
    diag_z = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(diag_z, 0.0)

    off_diag_text = [
        ["" if r == c else f"{c_vals[r, c]:.3f}" for c in range(n)]
        for r in range(n)
    ]
    diag_text = [
        ["-" if r == c else "" for c in range(n)]
        for r in range(n)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=main_z,
        x=labels,
        y=labels,
        text=off_diag_text,
        texttemplate="%{text}",
        textfont=dict(size=10, color="#1A1A1A"),
        colorscale=[[0, '#FF3B30'], [0.5, '#FF9F0A'], [1, '#34C759']],
        colorbar=dict(
            title=dict(text="Score", font=dict(color="#9B9B9B", size=10)),
            tickformat=".2f",
            tickfont=dict(size=9, color="#9B9B9B"),
            x=1.02,
            xanchor="left",
            thickness=12,
            len=0.8,
        ),
        hoverongaps=False,
    ))

    # Diagonal overlay: neutral warm-grey, no colorbar
    fig.add_trace(go.Heatmap(
        z=diag_z,
        x=labels,
        y=labels,
        text=diag_text,
        texttemplate="%{text}",
        textfont=dict(size=11, color="#9B9B9B"),
        colorscale=[[0.0, "#F6F2EC"], [1.0, "#F6F2EC"]],
        zmin=-0.5, zmax=0.5,
        showscale=False,
        hoverongaps=False,
        hovertemplate="<b>%{x}</b><br>Self-comparison<extra></extra>",
    ))

    fig.update_layout(**{k: v for k, v in plotly_layout.items() if k != 'title'})
    fig.update_layout(
        title=dict(text="Pairwise Competition Matrix", font=dict(size=16, color="#1D1D1F")),
        height=420,
        margin=dict(r=90),
        plot_bgcolor="#FFFFFF",
        showlegend=False,
    )
    return fig


def chart_scores_bar(scores_df: pd.DataFrame, colours: Dict[str, str], plotly_layout: Dict, title: str) -> go.Figure:
    fig = go.Figure()
    for idx in scores_df.index:
        score_pct = scores_df.loc[idx, 'Score'] * 100
        fig.add_trace(go.Bar(
            x=[idx],
            y=[score_pct],
            name=idx,
            marker_color=colours.get(idx, '#0071E3'),
            text=f"{score_pct:.1f}%",
            textposition='auto'
        ))
    fig.update_layout(**{k: v for k, v in plotly_layout.items() if k != 'title'})
    fig.update_layout(title=dict(text=title, font=dict(size=16, color="#1D1D1F")))
    fig.update_xaxes(title_text="Fund")
    fig.update_yaxes(title_text="TOPSIS Score (%)")
    return fig


def chart_sensitivity_heatmap(score_data: pd.DataFrame, text_data: pd.DataFrame, plotly_layout: Dict) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=score_data.values,
        x=score_data.columns,
        y=score_data.index,
        colorscale=[[0, '#FF3B30'], [0.5, '#FF9F0A'], [1, '#34C759']],
        text=text_data.values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(
            title=dict(text="Score", font=dict(color="#9B9B9B", size=10)),
            tickformat=".1%",
            tickfont=dict(size=9, color="#9B9B9B"),
            x=1.02,
            xanchor="left",
            thickness=12,
            len=0.8,
        ),
    ))
    fig.update_layout(**{k: v for k, v in plotly_layout.items() if k != 'title'})
    fig.update_layout(
        title=dict(text="Yuan & Yuan Scores Across Weighting Schemes", font=dict(size=16, color="#1D1D1F")),
        height=max(200, score_data.shape[0] * 55 + 70),
        margin=dict(t=50, b=20, r=110),
    )
    return fig