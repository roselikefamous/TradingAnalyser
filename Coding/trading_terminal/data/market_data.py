from typing import Any, Optional

import streamlit as st
import yfinance as yf

from trading_terminal.contracts import MarketDataBundle
from trading_terminal.indicators import apply_core_indicators


@st.cache_data(ttl=120)
def fetch_market_data(ticker: str, period: str, interval: str) -> Optional[MarketDataBundle]:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        if df.empty:
            return None

        info: dict[str, Any]
        news: list[dict[str, Any]]
        dividends = None
        earnings_dates = None

        try:
            info = t.info
        except Exception:
            info = {}

        try:
            news = t.news
        except Exception:
            news = []

        try:
            dividends = t.dividends
        except Exception:
            dividends = None

        try:
            earnings_dates = t.get_earnings_dates(limit=5)
        except Exception:
            earnings_dates = None

        return MarketDataBundle(
            df=df,
            info=info,
            news=news,
            dividends=dividends,
            earnings_dates=earnings_dates,
        )
    except Exception:
        return None


def apply_indicators(df, ema1_len: int, ema2_len: int, rsi_len: int):
    return apply_core_indicators(df, ema1_len, ema2_len, rsi_len)
