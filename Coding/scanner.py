"""
scanner.py – Multi-Asset Market Scanner Engine
Analyzes all predefined assets and ranks them by composite indicator score.
"""

from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf
from indicators import (
    apply_core_indicators, calc_ema, calc_rsi, calc_macd,
    calc_bollinger_bands, calc_atr, calc_adx, calc_stochastic,
    detect_hammer, detect_bullish_engulfing, detect_harami,
    detect_doji, detect_spinning_top
)
from strategy_engine import run_strategies


# ═══════════════════════════════════════════════════════════
# PREDEFINED ASSET UNIVERSE
# ═══════════════════════════════════════════════════════════

PREDEFINED_ASSETS = {
    # Tech & US Blue Chips
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft",
    "GOOG": "Alphabet (Google)",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "META": "Meta Platforms",
    "AMD": "Advanced Micro Devices",
    "NFLX": "Netflix",
    "INTC": "Intel",
    # DAX & Europe
    "SAP.DE": "SAP SE",
    "SIE.DE": "Siemens",
    "ALV.DE": "Allianz",
    "BMW.DE": "BMW",
    "MBG.DE": "Mercedes",
    "VOW3.DE": "Volkswagen",
    "RHM.DE": "Rheinmetall",
    # Crypto
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    # Indices & ETFs
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "DIA": "Dow Jones ETF",
    # Commodities
    "GC=F": "Gold",
    "SI=F": "Silver",
    "CL=F": "Crude Oil",
}


# ═══════════════════════════════════════════════════════════
# SCORING ENGINE
# ═══════════════════════════════════════════════════════════

def score_asset(symbol: str, name: str, strategies: list) -> Optional[dict]:
    """
    Downloads data for a single asset and computes a composite score 0-100
    using the AI Strategy Engine and Multi-Timeframe (MTF) confluence.
    """
    try:
        # Multi-Timeframe Downloads
        # 1. Daily Chart for Trend Context
        raw_1d = yf.Ticker(symbol).history(period="3mo", interval="1d")
        if raw_1d is None or raw_1d.empty or len(raw_1d) < 30:
            return None
            
        # 2. Hourly Chart for Entry Signals
        raw_1h = yf.Ticker(symbol).history(period="1mo", interval="1h")
        if raw_1h is None or raw_1h.empty or len(raw_1h) < 30:
            return None
            
        df_1d = apply_core_indicators(raw_1d)
        df_1h = apply_core_indicators(raw_1h)
        
        # ── MTF Trend Filter (Day Chart) ──
        last_1d = df_1d.iloc[-1]
        daily_close = last_1d["Close"]
        daily_ema21 = last_1d.get("EMA_2", np.nan)
        daily_ema55 = last_1d.get("EMA_55", np.nan)
        
        daily_trend = "NEUTRAL"
        if not np.isnan(daily_ema21) and not np.isnan(daily_ema55):
            if daily_close > daily_ema21 > daily_ema55:
                daily_trend = "BULL"
            elif daily_close < daily_ema21 < daily_ema55:
                daily_trend = "BEAR"

        # ── AI Strategy Engine (Hour Chart) ──
        engine_res = run_strategies(df_1h, strategies)
        
        raw_action = engine_res['action']
        score = engine_res['score']
        matching_books = engine_res['matching_books']
        
        # ── Validate MTF Confluence ──
        # Don't buy on the 1H if the 1D trend is bearish
        if raw_action == "BUY" and daily_trend == "BEAR":
            direction = "🟡 NEUTRAL (MTF Block)"
            direction_key = "NEUTRAL"
            signals = ["1H Buy Blocked by 1D Bear Trend"]
            score = score * 0.5  # Penalize score
        elif raw_action == "SELL" and daily_trend == "BULL":
            direction = "🟡 NEUTRAL (MTF Block)"
            direction_key = "NEUTRAL"
            signals = ["1H Sell Blocked by 1D Bull Trend"]
            score = score * 0.5
        else:
            if raw_action == "BUY":
                direction = "🟢 KAUFEN"
                direction_key = "BUY"
            elif raw_action == "SELL":
                direction = "🔴 VORSICHT"
                direction_key = "SELL"
            else:
                direction = "🟡 NEUTRAL"
                direction_key = "NEUTRAL"
                
            signals = [f"✅ {s[0]} ({s[1]})" for s in matching_books] if matching_books else []
            if daily_trend == "BULL" and raw_action == "BUY":
                signals.append("✅ MTF Confluence (1D Bull)")
                score = min(100, score + 10) # Bonus for MTF alignment

        last_1h = df_1h.iloc[-1]
        close = last_1h["Close"]
        rsi = last_1h.get("RSI", np.nan)
        adx = last_1h.get("ADX", np.nan)
        atr = last_1h.get("ATR", close * 0.02)
        if np.isnan(atr): atr = close * 0.02

        entry = close
        sl = 0.0
        tp = 0.0
        if raw_action == "BUY":
            sl = entry - (atr * 1.5)
            tp = entry + (atr * 2.5)
        elif raw_action == "SELL":
            sl = entry + (atr * 1.5)
            tp = entry - (atr * 2.5)

        perf_1d = (df_1d["Close"].iloc[-1] / df_1d["Close"].iloc[-2] - 1) * 100 if len(df_1d) >= 2 else 0
        perf_5d = (df_1d["Close"].iloc[-1] / df_1d["Close"].iloc[-6] - 1) * 100 if len(df_1d) >= 6 else 0

        return {
            "Symbol": symbol,
            "Name": name,
            "Score": round(score, 1),
            "Direction": direction,
            "DirectionKey": direction_key,
            "Kurs": round(close, 2),
            "Entry": round(entry, 2),
            "SL": round(sl, 2),
            "TP": round(tp, 2),
            "1T %": round(perf_1d, 2),
            "5T %": round(perf_5d, 2),
            "RSI": round(rsi, 1) if not np.isnan(rsi) else None,
            "ADX": round(adx, 1) if not np.isnan(adx) else None,
            "Pattern": f"Trend: {daily_trend}",
            "Signale": " | ".join(signals) if signals else "Keine Buch-Signale aktiv",
            "EMA_Layers": 0, # Legacy field
        }

    except Exception as e:
        print(f"Error scanning {symbol}: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# MAIN SCAN FUNCTION
# ═══════════════════════════════════════════════════════════

def scan_all_assets(assets: dict = None, progress_callback=None) -> pd.DataFrame:
    """
    Scan all assets and return a DataFrame sorted by Score (best first).

    Args:
        assets: dict of {symbol: name}. Defaults to PREDEFINED_ASSETS.
        progress_callback: optional callable(current, total, symbol) for progress updates.

    Returns:
        DataFrame sorted by Score descending.
    """
    if assets is None:
        assets = PREDEFINED_ASSETS

    results = []
    total = len(assets)

    import database
    active_strategies = database.get_all_strategies()

    for i, (symbol, name) in enumerate(assets.items()):
        if progress_callback:
            progress_callback(i, total, symbol)

        result = score_asset(symbol, name, active_strategies)
        if result:
            results.append(result)

    if progress_callback:
        progress_callback(total, total, "Fertig")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # rank starts at 1
    return df
