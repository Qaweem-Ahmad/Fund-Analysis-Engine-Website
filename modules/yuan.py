import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class YuanYuan:
    def __init__(self, df: pd.DataFrame, benefits: List[str], costs: List[str]):
        self.df = df.copy()
        self.benefits = benefits
        self.costs = costs
        self.pillar_to_metrics = {
            'Returns': ['Ann. Return (%)', 'Alpha (ann. %)', 'Upside Capture (%)', 'Calmar Ratio'],
            'Risk-Adj': ['Sharpe Ratio', 'Sortino Ratio', 'Treynor Ratio', 'Information Ratio', 'R²'],
            'Risk/DD': ['Ann. Volatility (%)', 'Beta', 'Tracking Error (%)', 'Max Drawdown (%)', 'Max DD Duration (mths)', 'Downside Capture (%)'],
            'Costs': ['OCF'],
            'ESG': ['ESG Globe Rating', 'Carbon Risk Score']
        }

    def _get_weights(self, pillar_weights: Dict[str, float]) -> Dict[str, float]:
        raw = {}
        for pillar, metrics in self.pillar_to_metrics.items():
            w = pillar_weights[pillar] / len(metrics)
            for m in metrics:
                raw[m] = w
        # Only weight metrics that are actually present in the dataframe
        active = {m: w for m, w in raw.items() if m in self.df.index}
        total = sum(active.values())
        if total <= 0:
            raise ValueError("No active metrics found in dataframe — weights sum to 0.")
        return {m: w / total for m, w in active.items()}

    def competition_matrix(self, pillar_weights: Dict[str, float]) -> pd.DataFrame:
        weights = self._get_weights(pillar_weights)
        funds = self.df.columns
        c_matrix = pd.DataFrame(index=funds, columns=funds, dtype=float)
        for i in funds:
            for j in funds:
                score = 0.0
                for k in weights:   # iterate only over weighted (active) metrics
                    w = weights[k]
                    val_i = self.df.loc[k, i]
                    val_j = self.df.loc[k, j]
                    if k in self.benefits:
                        if val_i > val_j:
                            score += w
                        elif val_i == val_j:
                            score += 0.5 * w
                    elif k in self.costs:
                        if val_i < val_j:
                            score += w
                        elif val_i == val_j:
                            score += 0.5 * w
                c_matrix.loc[i, j] = score
        return c_matrix

    def power_iteration(self, c_matrix: pd.DataFrame, n_iter: int = 1000, tol: float = 1e-9) -> Tuple[np.ndarray, int]:
        n = len(c_matrix)
        v = np.ones(n) / n
        for step in range(n_iter):
            v_new = c_matrix.values @ v
            norm = np.linalg.norm(v_new)
            if norm > 0:
                v_new = v_new / norm
            if np.linalg.norm(v_new - v) < tol:
                return v_new, step + 1
            v = v_new
        return v, n_iter

    def rank(self, scores: np.ndarray) -> pd.DataFrame:
        scores_series = pd.Series(scores, index=self.df.columns)
        ranking = scores_series.sort_values(ascending=False)
        ranking_df = pd.DataFrame({'Score': ranking, 'Rank': range(1, len(ranking) + 1)})
        return ranking_df

    def run(self, pillar_weights: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        c_matrix = self.competition_matrix(pillar_weights)
        scores, n = self.power_iteration(c_matrix)
        # Normalise to sum-to-1 (L1) so scores are interpretable as proportions
        if scores.sum() > 0:
            scores = scores / scores.sum()
        ranking = self.rank(scores)
        return ranking, c_matrix, n