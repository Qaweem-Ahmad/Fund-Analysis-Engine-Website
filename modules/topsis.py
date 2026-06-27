import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class TOPSIS:
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

    def normalise(self) -> pd.DataFrame:
        """Vector normalization: r_ij = x_ij / sqrt(sum x_ij^2) — norm per metric row"""
        norm_df = self.df.div(np.sqrt((self.df ** 2).sum(axis=1)), axis=0)
        return norm_df

    def weight(self, pillar_weights: Dict[str, float]) -> pd.DataFrame:
        """Apply weights distributed by pillars"""
        weights = {}
        for pillar, metrics in self.pillar_to_metrics.items():
            w = pillar_weights[pillar] / len(metrics)
            for m in metrics:
                weights[m] = w
        weighted_df = self.normalise().mul(pd.Series(weights), axis=0)
        return weighted_df

    def ideal_solutions(self, weighted_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Compute ideal positive and negative solutions"""
        A_plus = pd.Series(index=weighted_df.index)
        A_minus = pd.Series(index=weighted_df.index)
        for metric in weighted_df.index:
            if metric in self.benefits:
                A_plus[metric] = weighted_df.loc[metric].max()
                A_minus[metric] = weighted_df.loc[metric].min()
            elif metric in self.costs:
                A_plus[metric] = weighted_df.loc[metric].min()
                A_minus[metric] = weighted_df.loc[metric].max()
        return A_plus, A_minus

    def distances(self, weighted_df: pd.DataFrame, A_plus: pd.Series, A_minus: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Compute distances to ideal solutions — align Series by row index (metric), not column"""
        D_plus = np.sqrt((weighted_df.sub(A_plus, axis=0) ** 2).sum(axis=0))
        D_minus = np.sqrt((weighted_df.sub(A_minus, axis=0) ** 2).sum(axis=0))
        return D_plus, D_minus

    def score(self, D_plus: pd.Series, D_minus: pd.Series) -> pd.Series:
        """Compute TOPSIS scores: S_i = D- / (D+ + D-). Identical funds give D+=D-=0; assign 0.5."""
        denom = D_plus + D_minus
        scores = D_minus / denom
        scores = scores.where(denom != 0, 0.5)
        return scores

    def rank(self, scores: pd.Series) -> pd.DataFrame:
        """Rank funds by scores descending"""
        ranking = scores.sort_values(ascending=False)
        ranking_df = pd.DataFrame({'Score': ranking, 'Rank': range(1, len(ranking) + 1)})
        return ranking_df

    def run(self, pillar_weights: Dict[str, float]) -> pd.DataFrame:
        """Run full TOPSIS analysis"""
        weighted_df = self.weight(pillar_weights)
        A_plus, A_minus = self.ideal_solutions(weighted_df)
        D_plus, D_minus = self.distances(weighted_df, A_plus, A_minus)
        scores = self.score(D_plus, D_minus)
        ranking = self.rank(scores)
        return ranking