from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd

class DataProviderInterface(ABC):
    """Abstract base class for all market data providers."""

    @abstractmethod
    def fetch_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetches historical price data."""
        pass

    @abstractmethod
    def fetch_info(self, symbol: str) -> dict[str, Any]:
        """Fetches general metadata about an asset."""
        pass

    @abstractmethod
    def fetch_news(self, symbol: str) -> list[dict[str, Any]]:
        """Fetches recent news for an asset."""
        pass

    @abstractmethod
    def fetch_dividends(self, symbol: str) -> Optional[pd.Series]:
        """Fetches dividend history."""
        pass

    @abstractmethod
    def fetch_earnings_dates(self, symbol: str, limit: int = 5) -> Optional[pd.DataFrame]:
        """Fetches upcoming or past earnings dates."""
        pass
