import requests
import pandas as pd
from typing import Any, Optional
from trading_terminal.data.base import DataProviderInterface
from trading_terminal.data.yf_provider import YFinanceProvider
from trading_terminal.config import settings
from trading_terminal.utils.logger import structured_logger

class Trading212Provider(DataProviderInterface):
    """Implementation of DataProviderInterface using Trading 212 REST API (Beta)."""

    def __init__(self):
        self.api_key = settings.t212_api_key
        self.base_url = "https://live.trading212.com/api/v0" # Default to live
        self.headers = {"Authorization": self.api_key} if self.api_key else {}
        self._yf_fallback = YFinanceProvider()

    def fetch_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Trading 212 REST API is account-focused; use yfinance fallback for OHLCV.
        """
        structured_logger.info(
            f"T212: fetch_history for {symbol} delegated to yfinance fallback "
            f"(period={period}, interval={interval})."
        )
        return self._yf_fallback.fetch_history(symbol, period, interval)

    def fetch_info(self, symbol: str) -> dict[str, Any]:
        """Fetches asset info from T212 (Beta)."""
        if not self.api_key:
            return {}
        try:
            # T212 endpoint for metadata (example)
            # Search for the instrument first
            url = f"{self.base_url}/equity/metadata/instruments"
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                instruments = response.json()
                # Find the specific instrument
                for inst in instruments:
                    if inst.get('ticker') == symbol:
                        return {
                            "shortName": inst.get('shortName'),
                            "full_name": inst.get('fullName'),
                            "isin": inst.get('isin'),
                            "currency": inst.get('currencyCode'),
                            "status": "active"
                        }
            return {}
        except Exception as e:
            structured_logger.error(f"T212 fetch_info failed for {symbol}: {str(e)}")
            return {}

    def fetch_news(self, symbol: str) -> list[dict[str, Any]]:
        return []

    def fetch_dividends(self, symbol: str) -> Optional[pd.Series]:
        return None

    def fetch_earnings_dates(self, symbol: str, limit: int = 5) -> Optional[pd.DataFrame]:
        return None

    def fetch_account_summary(self) -> dict[str, Any]:
        """Specific T212 method for account cash/portfolio."""
        if not self.api_key:
            return {}
        try:
            url = f"{self.base_url}/equity/account/cash"
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            structured_logger.error(f"T212 fetch_account_summary failed: {str(e)}")
            return {}
