import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Tuple


def fetch_data(tickers: List[str], benchmark: str, start: str, end: str) -> pd.DataFrame:
    """
    Download monthly price data via yfinance. Returns monthly log returns DataFrame.
    """
    all_tickers = tickers + [benchmark]
    data = yf.download(all_tickers, start=start, end=end, interval='1mo', auto_adjust=True, progress=False)['Close']
    data = data.dropna()  # Drop rows with NaN
    log_returns = compute_log_returns(data)
    return log_returns


def compute_log_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly log returns: ln(P_t / P_{t-1})
    """
    return np.log(monthly_prices / monthly_prices.shift(1)).dropna()