import os
from typing import Any, Optional

import streamlit as st

from trading_terminal.contracts import MarketDataBundle
from trading_terminal.indicators import apply_core_indicators
from trading_terminal.data.yf_provider import YFinanceProvider
from trading_terminal.data.alpaca_provider import AlpacaProvider
from trading_terminal.config import settings


def get_provider():
    """Simple factory to return the configured data provider."""
    provider_name = settings.data_provider
    
    if provider_name == "alpaca":
        return AlpacaProvider()
    return YFinanceProvider()


from trading_terminal.utils.logger import structured_logger

@st.cache_data(ttl=120)
def fetch_market_data(ticker: str, period: str, interval: str) -> Optional[MarketDataBundle]:
    try:
        provider = get_provider()
        
        df = provider.fetch_history(ticker, period, interval)
        if df is None or df.empty:
            structured_logger.warning(f"No data returned for ticker: {ticker}")
            return None

        info = provider.fetch_info(ticker)
        news = provider.fetch_news(ticker)
        dividends = provider.fetch_dividends(ticker)
        earnings_dates = provider.fetch_earnings_dates(ticker)

        return MarketDataBundle(
            df=df,
            info=info,
            news=news,
            dividends=dividends,
            earnings_dates=earnings_dates,
        )
    except Exception as e:
        structured_logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def apply_indicators(df, ema1_len: int, ema2_len: int, rsi_len: int):
    return apply_core_indicators(df, ema1_len, ema2_len, rsi_len)
