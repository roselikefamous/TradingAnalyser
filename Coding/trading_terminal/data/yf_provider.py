import yfinance as yf
import pandas as pd
from typing import Any, Optional
from trading_terminal.data.base import DataProviderInterface

class YFinanceProvider(DataProviderInterface):
    """Implementation of DataProviderInterface using yfinance."""

    def fetch_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        try:
            t = yf.Ticker(symbol)
            df = t.history(period=period, interval=interval)
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception:
            return pd.DataFrame()

    def fetch_info(self, symbol: str) -> dict[str, Any]:
        try:
            t = yf.Ticker(symbol)
            return t.info
        except Exception:
            return {}

    def fetch_news(self, symbol: str) -> list[dict[str, Any]]:
        try:
            t = yf.Ticker(symbol)
            return t.news
        except Exception:
            return []

    def fetch_dividends(self, symbol: str) -> Optional[pd.Series]:
        try:
            t = yf.Ticker(symbol)
            return t.dividends
        except Exception:
            return None

    def fetch_earnings_dates(self, symbol: str, limit: int = 5) -> Optional[pd.DataFrame]:
        try:
            t = yf.Ticker(symbol)
            return t.get_earnings_dates(limit=limit)
        except Exception:
            return None
