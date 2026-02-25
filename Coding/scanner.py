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

def score_asset(symbol: str, name: str) -> Optional[dict]:
    """
    Downloads data for a single asset and computes a composite score 0-100.

    Score breakdown:
      Trend (EMA stack)      0-30 pts
      RSI momentum zone      0-15 pts
      MACD alignment         0-15 pts
      Bollinger Band setup   0-10 pts
      ADX trend strength     0-10 pts
      Stochastic crossover   0-10 pts
      Candlestick pattern    0-10 pts

    Returns a dict with all relevant signal fields, or None on error.
    """
    try:
        raw = yf.Ticker(symbol).history(period="3mo", interval="1d")
        if raw is None or raw.empty or len(raw) < 30:
            return None
    except Exception:
        return None

    try:
        df = apply_core_indicators(raw)
        last = df.iloc[-1]
        close = last["Close"]

        score = 0
        signals = []

        # ── 1. TREND SCORE (0-30) ──────────────────────────
        trend_score = 0
        bullish_layers = 0

        ema9  = last.get("EMA_1", np.nan)   # configured as EMA_1 (short)
        ema21 = last.get("EMA_2", np.nan)   # configured as EMA_2 (long)
        ema55 = last.get("EMA_55", np.nan)
        sma200 = last.get("SMA_200", np.nan)

        if not np.isnan(ema9)   and close > ema9:   bullish_layers += 1
        if not np.isnan(ema21)  and close > ema21:  bullish_layers += 1
        if not np.isnan(ema55)  and close > ema55:  bullish_layers += 1
        if not np.isnan(sma200) and close > sma200: bullish_layers += 1
        # Golden cross: EMA9 > EMA21
        if not np.isnan(ema9) and not np.isnan(ema21) and ema9 > ema21:
            bullish_layers += 1

        trend_score = bullish_layers * 6  # max 5 layers * 6 = 30
        score += trend_score
        if bullish_layers >= 4:
            signals.append("✅ Starker Aufwärtstrend (EMA-Stack)")
        elif bullish_layers >= 2:
            signals.append("➡️ Moderater Trend")

        # ── 2. RSI SCORE (0-15) ────────────────────────────
        rsi_score = 0
        rsi = last.get("RSI", np.nan)
        rsi_label = f"RSI {rsi:.0f}" if not np.isnan(rsi) else "RSI n/a"

        if not np.isnan(rsi):
            if 50 <= rsi <= 65:          # sweet spot: bullish momentum
                rsi_score = 15
                signals.append("✅ RSI Bullisch (50–65)")
            elif 40 <= rsi < 50:         # neutral-bullish
                rsi_score = 10
            elif 30 <= rsi < 40:         # approaching oversold
                rsi_score = 12
                signals.append("⚡ RSI Überverkauft-Zone nahe")
            elif rsi < 30:               # oversold = bounce potential
                rsi_score = 13
                signals.append("⚡ RSI Überverkauft (<30)")
            elif 65 < rsi <= 75:         # getting extended
                rsi_score = 5
            else:                        # >75 = very overbought
                rsi_score = 2
        score += rsi_score

        # ── 3. MACD SCORE (0-15) ───────────────────────────
        macd_score = 0
        macd  = last.get("MACD", np.nan)
        signal_line = last.get("Signal", np.nan)
        hist = last.get("MACD_Hist", np.nan)
        prev_hist = df["MACD_Hist"].iloc[-2] if len(df) >= 2 else np.nan

        if not np.isnan(macd) and not np.isnan(signal_line):
            if macd > signal_line:
                macd_score += 7
                if not np.isnan(hist) and not np.isnan(prev_hist) and hist > prev_hist:
                    macd_score += 8  # histogram accelerating
                    signals.append("✅ MACD Bullisch & Beschleunigend")
                else:
                    signals.append("✅ MACD über Signal")
        score += macd_score

        # ── 4. BOLLINGER BAND SCORE (0-10) ─────────────────
        bb_score = 0
        bb_lower = last.get("BB_Lower", np.nan)
        bb_upper = last.get("BB_Upper", np.nan)
        bb_mid   = last.get("BB_Mid", np.nan)

        if not np.isnan(bb_lower) and not np.isnan(bb_upper) and not np.isnan(bb_mid):
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                # Near lower band (bounce setup)
                dist_lower = (close - bb_lower) / bb_range
                if dist_lower < 0.15:
                    bb_score = 10
                    signals.append("⚡ Kurs am unteren Bollinger Band")
                elif dist_lower < 0.35:
                    bb_score = 6
                    signals.append("➡️ Kurs unterhalb BB-Mitte")
                elif dist_lower > 0.85:
                    bb_score = 2  # at upper band (overbought risk)
                else:
                    bb_score = 4
        score += bb_score

        # ── 5. ADX TREND STRENGTH (0-10) ───────────────────
        adx_score = 0
        adx = last.get("ADX", np.nan)
        if not np.isnan(adx):
            if adx >= 40:
                adx_score = 10
                signals.append("✅ ADX Starker Trend (>40)")
            elif adx >= 25:
                adx_score = 7
                signals.append("✅ ADX Trend (>25)")
            elif adx >= 15:
                adx_score = 3
        score += adx_score

        # ── 6. STOCHASTIC CROSSOVER (0-10) ─────────────────
        stoch_score = 0
        stoch_k = last.get("Stoch_K", np.nan)
        stoch_d = last.get("Stoch_D", np.nan)
        prev_k  = df["Stoch_K"].iloc[-2] if len(df) >= 2 else np.nan
        prev_d  = df["Stoch_D"].iloc[-2] if len(df) >= 2 else np.nan

        if not np.isnan(stoch_k) and not np.isnan(stoch_d):
            # %K crossed above %D from oversold zone
            if (not np.isnan(prev_k) and not np.isnan(prev_d)
                    and prev_k < prev_d and stoch_k > stoch_d and stoch_k < 50):
                stoch_score = 10
                signals.append("✅ Stochastik Bullischer Kreuz (überverkauft)")
            elif stoch_k < 20:
                stoch_score = 6
                signals.append("⚡ Stochastik Überverkauft (<20)")
            elif 40 < stoch_k < 60 and stoch_k > stoch_d:
                stoch_score = 4
        score += stoch_score

        # ── 7. CANDLESTICK PATTERN (0-10) ──────────────────
        pattern_score = 0
        pattern_name = ""
        # Check last 3 candles for patterns
        recent_df = df.iloc[-3:]
        hammers    = detect_hammer(recent_df)
        engulfings = detect_bullish_engulfing(recent_df)
        haramis    = detect_harami(recent_df)
        dojis      = detect_doji(recent_df)

        if engulfings.any():
            pattern_score = 10
            pattern_name = "Bullish Engulfing"
        elif hammers.any():
            pattern_score = 8
            pattern_name = "Hammer"
        elif haramis.any():
            pattern_score = 7
            pattern_name = "Harami"
        elif dojis.any():
            pattern_score = 4
            pattern_name = "Doji"

        if pattern_name:
            signals.append(f"✅ Pattern: {pattern_name}")
        score += pattern_score

        # ── DIRECTION LABEL ─────────────────────────────────
        score = min(100, max(0, score))

        if score >= 65:
            direction = "🟢 KAUFEN"
            direction_key = "BUY"
        elif score >= 45:
            direction = "🟡 NEUTRAL"
            direction_key = "NEUTRAL"
        else:
            direction = "🔴 VORSICHT"
            direction_key = "SELL"

        # 1-day and 5-day performance
        perf_1d = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100 if len(df) >= 2 else 0
        perf_5d = (df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100 if len(df) >= 6 else 0

        return {
            "Symbol": symbol,
            "Name": name,
            "Score": round(score, 1),
            "Direction": direction,
            "DirectionKey": direction_key,
            "Kurs": round(close, 2),
            "1T %": round(perf_1d, 2),
            "5T %": round(perf_5d, 2),
            "RSI": round(rsi, 1) if not np.isnan(rsi) else None,
            "ADX": round(adx, 1) if not np.isnan(adx) else None,
            "Pattern": pattern_name or "–",
            "Signale": " | ".join(signals) if signals else "Keine klaren Signale",
            "EMA_Layers": bullish_layers,
        }

    except Exception as e:
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

    for i, (symbol, name) in enumerate(assets.items()):
        if progress_callback:
            progress_callback(i, total, symbol)

        result = score_asset(symbol, name)
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
