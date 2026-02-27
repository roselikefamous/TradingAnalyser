import pandas as pd
from typing import Any, Optional
from trading_terminal.data.base import DataProviderInterface

class AlpacaProvider(DataProviderInterface):
    """Stub implementation for Alpaca Data Provider."""

    def fetch_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        # This would use alpaca-trade-api-python
        print(f"[RESERVED] Alpaca fetching history for {symbol}")
        return pd.DataFrame()

    def fetch_info(self, symbol: str) -> dict[str, Any]:
        return {"provider": "Alpaca", "status": "Stub"}

    def fetch_news(self, symbol: str) -> list[dict[str, Any]]:
        return []

    def fetch_dividends(self, symbol: str) -> Optional[pd.Series]:
        return None

    def fetch_earnings_dates(self, symbol: str, limit: int = 5) -> Optional[pd.DataFrame]:
        return None
