import pandas as pd
import alpaca_trade_api as tradeapi
from typing import Any, Optional
from trading_terminal.data.base import DataProviderInterface
from trading_terminal.config import settings
from trading_terminal.utils.logger import structured_logger

class AlpacaProvider(DataProviderInterface):
    """Implementation of DataProviderInterface using Alpaca Trade API."""

    def __init__(self):
        self.api_key = settings.alpaca_api_key
        self.secret_key = settings.alpaca_secret_key
        self.base_url = settings.alpaca_base_url
        
        if not self.api_key or not self.secret_key:
            structured_logger.error("Alpaca API credentials missing in settings.")
            self.api = None
        else:
            self.api = tradeapi.REST(self.api_key, self.secret_key, self.base_url, api_version='v2')

    def fetch_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        if not self.api:
            return pd.DataFrame()
            
        # Map period/interval to Alpaca timeframe
        # Note: This is a simplified mapping for illustration
        timeframe = tradeapi.TimeFrame.Day if "d" in interval else tradeapi.TimeFrame.Hour
        
        try:
            bars = self.api.get_bars(symbol, timeframe, limit=1000).df
            if bars.empty:
                return bars
            
            # Standardize column names to match the app's expectation (from yfinance style)
            bars = bars.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            })
            return bars
        except Exception as e:
            structured_logger.error(f"Alpaca fetch_history failed for {symbol}: {str(e)}")
            return pd.DataFrame()

    def fetch_info(self, symbol: str) -> dict[str, Any]:
        if not self.api:
            return {}
        try:
            asset = self.api.get_asset(symbol)
            return {
                "shortName": asset.name,
                "sector": asset.class_,
                "exchange": asset.exchange,
                "status": asset.status
            }
        except Exception as e:
            structured_logger.error(f"Alpaca fetch_info failed for {symbol}: {str(e)}")
            return {}

    def fetch_news(self, symbol: str) -> list[dict[str, Any]]:
        # Alpaca news API is separate but for now return empty
        return []

    def fetch_dividends(self, symbol: str) -> Optional[pd.Series]:
        return None

    def fetch_earnings_dates(self, symbol: str, limit: int = 5) -> Optional[pd.DataFrame]:
        return None
