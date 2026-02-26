"""
strategies.py – Perfect Trading Strategy Logic
Three strategies with full backtesting capabilities.
"""
import numpy as np
import pandas as pd
from indicators import (
    calc_ema, calc_sma, calc_bollinger_bands, calc_rsi, calc_vwap,
    calc_stochastic, calc_atr, calc_fibonacci_levels, detect_hammer,
    detect_doji, detect_spinning_top, detect_bullish_engulfing,
    detect_bearish_engulfing, detect_shooting_star, detect_harami, find_swing_points
)


# ═══════════════════════════════════════════════════════════
# UNIVERSAL: Position Size Calculator
# ═══════════════════════════════════════════════════════════

def calc_position_size(capital: float, risk_pct: float, entry_price: float, sl_price: float):
    """Calculate position size based on max risk rule (e.g., 2% of capital).

    Returns:
        dict with max_loss, sl_distance, shares, investment_volume, risk_reward_note
    """
    max_loss = capital * (risk_pct / 100)
    sl_distance = abs(entry_price - sl_price)
    shares = max_loss / sl_distance if sl_distance > 0 else 0
    investment = shares * entry_price

    return {
        'max_loss': max_loss,
        'sl_distance': sl_distance,
        'shares': int(shares),
        'investment': investment,
        'risk_pct_actual': risk_pct,
        'warning': investment > capital,
    }


# ═══════════════════════════════════════════════════════════
# STRATEGY 1: BOLLINGER BAND SCALPING
# ═══════════════════════════════════════════════════════════

def strategy_bollinger_scalping(df: pd.DataFrame):
    """Bollinger Band Scalping Strategy.

    Rules:
    - Trend filter: EMA 55 must be rising (bullish) → LONG only
    - Entry: Price touches/pierces lower BB → wait for bullish candle → Stop-Buy above its High
    - Stop-Loss: Below the significant low of last 3 candles
    - Take Profit: Trailing at low of 3rd-from-last candle, or fixed target at BB Mid
    - Break-even: After +10 pips, move SL to entry
    - Confluence: Stochastic %K < 20 (oversold) preferred

    Returns:
        entries: list of dicts with Date, Entry, SL, TP, Type, StochConfirm
        exits: list of dicts with Date, Price, Reason
        backtest: dict with stats
    """
    # Calculate required indicators
    bb_mid, bb_upper, bb_lower = calc_bollinger_bands(df['Close'], 20, 2)
    ema_55 = calc_ema(df['Close'], 55)
    stoch_k, stoch_d = calc_stochastic(df)

    df_s = df.copy()
    df_s['BB_Mid'] = bb_mid
    df_s['BB_Upper'] = bb_upper
    df_s['BB_Lower'] = bb_lower
    df_s['EMA_55'] = ema_55
    df_s['Trend_Up'] = ema_55.diff() > 0
    df_s['Stoch_K'] = stoch_k

    entries = []
    exits = []
    in_trade = False
    trade_entry_price = 0
    trade_sl = 0

    for i in range(5, len(df_s)):
        if pd.isna(df_s['BB_Lower'].iloc[i]) or pd.isna(df_s['EMA_55'].iloc[i]):
            continue

        if not in_trade:
            # Check conditions for entry
            is_uptrend = df_s['Trend_Up'].iloc[i - 1]
            touched_lower_bb = df_s['Low'].iloc[i - 1] <= df_s['BB_Lower'].iloc[i - 1]
            bullish_candle = df_s['Close'].iloc[i] > df_s['Open'].iloc[i]
            stoch_oversold = df_s['Stoch_K'].iloc[i] < 30 if not pd.isna(df_s['Stoch_K'].iloc[i]) else False

            if is_uptrend and touched_lower_bb and bullish_candle:
                entry_price = df_s['High'].iloc[i]  # Stop-Buy above high
                sl_price = min(df_s['Low'].iloc[max(0, i - 2):i + 1])  # Below recent lows
                tp_price = df_s['BB_Mid'].iloc[i]  # Target: BB middle

                entries.append({
                    'Date': df_s.index[i],
                    'Entry': entry_price,
                    'SL': sl_price,
                    'TP': tp_price,
                    'Type': 'LONG',
                    'StochConfirm': stoch_oversold,
                })
                in_trade = True
                trade_entry_price = entry_price
                trade_sl = sl_price
        else:
            # Check exit conditions
            # Trailing stop: low of 3rd-from-last candle
            trailing_sl = df_s['Low'].iloc[max(0, i - 3)] if i >= 3 else trade_sl

            # Update SL if trailing is higher (break-even logic)
            current_sl = max(trade_sl, trailing_sl)

            # Exit at BB Mid (take profit target)
            if df_s['High'].iloc[i] >= df_s['BB_Mid'].iloc[i]:
                exits.append({
                    'Date': df_s.index[i],
                    'Price': df_s['BB_Mid'].iloc[i],
                    'Reason': 'TP (BB Mid)',
                })
                in_trade = False
            # Exit if SL hit
            elif df_s['Low'].iloc[i] <= current_sl:
                exits.append({
                    'Date': df_s.index[i],
                    'Price': current_sl,
                    'Reason': 'SL Hit',
                })
                in_trade = False

    # Backtest stats
    backtest = _calc_backtest_stats(entries, exits, 'LONG')

    return entries, exits, backtest, df_s


# ═══════════════════════════════════════════════════════════
# STRATEGY 2: DAYTRADING TOP/BOTTOM REVERSAL
# ═══════════════════════════════════════════════════════════

def strategy_reversal(df: pd.DataFrame):
    """Daytrading Top & Bottom Reversal Strategy.

    Rules:
    - SHORT (Top Reversal): 5+ consecutive up candles + RSI > 90 + price at/above upper BB + Doji/Spinning Top
    - LONG (Bottom Reversal): 5+ consecutive down candles + RSI < 10 + price at/below lower BB + Doji/Spinning Top
    - Entry: Break of the Doji's low (SHORT) or high (LONG)
    - Stop-Loss: Absolute high (SHORT) or low (LONG)
    - Take Profit: Next support level, VWAP, or EMA 21

    Returns:
        short_signals, long_signals, backtest, df_s
    """
    bb_mid, bb_upper, bb_lower = calc_bollinger_bands(df['Close'], 20, 2)
    rsi = calc_rsi(df['Close'], 14)

    df_s = df.copy()
    df_s['BB_Upper'] = bb_upper
    df_s['BB_Lower'] = bb_lower
    df_s['BB_Mid'] = bb_mid
    df_s['RSI'] = rsi
    df_s['Doji'] = detect_doji(df)
    df_s['Spinning_Top'] = detect_spinning_top(df)

    # VWAP for take-profit reference
    if 'Volume' in df.columns:
        df_s['VWAP'] = calc_vwap(df)

    df_s['EMA_21'] = calc_ema(df['Close'], 21)

    short_signals = []
    long_signals = []
    exits = []

    for i in range(6, len(df_s)):
        if pd.isna(df_s['RSI'].iloc[i]) or pd.isna(df_s['BB_Upper'].iloc[i]):
            continue

        is_doji_or_spinning = df_s['Doji'].iloc[i] or df_s['Spinning_Top'].iloc[i]

        # ── SHORT SIGNAL (Top Reversal) ──
        five_up = all(df_s['Close'].iloc[i - j] > df_s['Open'].iloc[i - j] for j in range(1, 6))
        rsi_extreme_high = df_s['RSI'].iloc[i] > 90  # Extreme overbought (90, not 70/80)

        if (five_up
                and rsi_extreme_high
                and df_s['High'].iloc[i] >= df_s['BB_Upper'].iloc[i] * 0.998
                and is_doji_or_spinning):

            entry_price = df_s['Low'].iloc[i]  # Enter short below Doji low
            sl_price = df_s['High'].iloc[i]  # SL at absolute high

            # Take profit: VWAP, EMA 21, or BB Mid (whichever is closest below)
            tp_candidates = [df_s['BB_Mid'].iloc[i], df_s['EMA_21'].iloc[i]]
            if 'VWAP' in df_s.columns and not pd.isna(df_s['VWAP'].iloc[i]):
                tp_candidates.append(df_s['VWAP'].iloc[i])
            tp_price = max([t for t in tp_candidates if t < entry_price], default=entry_price * 0.98)

            short_signals.append({
                'Date': df_s.index[i],
                'Entry': entry_price,
                'SL': sl_price,
                'TP': tp_price,
                'Type': 'SHORT',
                'RSI': df_s['RSI'].iloc[i],
                'Pattern': 'Doji' if df_s['Doji'].iloc[i] else 'Spinning Top',
            })

        # ── LONG SIGNAL (Bottom Reversal) ──
        five_down = all(df_s['Close'].iloc[i - j] < df_s['Open'].iloc[i - j] for j in range(1, 6))
        rsi_extreme_low = df_s['RSI'].iloc[i] < 10  # Extreme oversold (10, not 20/30)

        if (five_down
                and rsi_extreme_low
                and df_s['Low'].iloc[i] <= df_s['BB_Lower'].iloc[i] * 1.002
                and is_doji_or_spinning):

            entry_price = df_s['High'].iloc[i]  # Enter long above Doji high
            sl_price = df_s['Low'].iloc[i]  # SL at absolute low

            tp_candidates = [df_s['BB_Mid'].iloc[i], df_s['EMA_21'].iloc[i]]
            if 'VWAP' in df_s.columns and not pd.isna(df_s['VWAP'].iloc[i]):
                tp_candidates.append(df_s['VWAP'].iloc[i])
            tp_price = min([t for t in tp_candidates if t > entry_price], default=entry_price * 1.02)

            long_signals.append({
                'Date': df_s.index[i],
                'Entry': entry_price,
                'SL': sl_price,
                'TP': tp_price,
                'Type': 'LONG',
                'RSI': df_s['RSI'].iloc[i],
                'Pattern': 'Doji' if df_s['Doji'].iloc[i] else 'Spinning Top',
            })

    all_signals = short_signals + long_signals
    backtest = _calc_backtest_stats_mixed(short_signals, long_signals, df_s)

    return short_signals, long_signals, backtest, df_s


# ═══════════════════════════════════════════════════════════
# STRATEGY 3: FIBONACCI SWING TRADING
# ═══════════════════════════════════════════════════════════

def strategy_fibonacci_swing(df: pd.DataFrame, lookback=60):
    """Fibonacci Correction Swing Trading Strategy.

    Rules:
    - Timeframe: Daily chart
    - Measure last major swing (highest high / lowest low)
    - Wait for correction to 50% or 61.8% Fibonacci level
    - Confirmation: Hammer, Bullish Engulfing, or Harami pattern at the level
    - Entry: Buy when price exceeds the previous day's high (confirmation)
    - Stop-Loss: Below the lowest low of the correction
    - Take Profit: Trailing stop under low of last 3-4 days
    - Extensions: 127.2% and 161.8% as targets

    Returns:
        fib_entries, backtest, df_s, fib_data
    """
    window = min(lookback, len(df) - 1)
    recent = df.iloc[-window:]

    swing_low = recent['Low'].min()
    swing_high = recent['High'].max()
    swing_low_idx = recent['Low'].idxmin()
    swing_high_idx = recent['High'].idxmax()

    retracements, extensions = calc_fibonacci_levels(swing_high, swing_low)
    fib_diff = swing_high - swing_low

    fib_500 = retracements['50.0%']
    fib_618 = retracements['61.8%']

    # Detect patterns
    hammers = detect_hammer(df)
    engulfings = detect_bullish_engulfing(df)
    haramis = detect_harami(df)

    fib_entries = []
    exits = []
    in_trade = False
    trade_entry = 0

    for i in range(2, len(df)):
        if fib_diff <= 0:
            break

        low = df['Low'].iloc[i]

        # Is price near 50% or 61.8% level?
        near_50 = abs(low - fib_500) / fib_diff < 0.025
        near_618 = abs(low - fib_618) / fib_diff < 0.025

        if (near_50 or near_618) and not in_trade:
            has_pattern = hammers.iloc[i] or engulfings.iloc[i] or haramis.iloc[i]

            if has_pattern:
                # Confirmation: price must exceed previous day's high
                if i + 1 < len(df) and df['High'].iloc[i + 1] > df['High'].iloc[i]:
                    entry_price = df['High'].iloc[i]  # Entry above pattern candle high
                    sl_price = min(df['Low'].iloc[max(0, i - 3):i + 1])  # Below correction low

                    # Determine which pattern was found
                    pattern = 'Hammer' if hammers.iloc[i] else 'Engulfing' if engulfings.iloc[i] else 'Harami'

                    fib_entries.append({
                        'Date': df.index[i],
                        'Entry': entry_price,
                        'SL': sl_price,
                        'Level': '50.0%' if near_50 else '61.8%',
                        'Pattern': pattern,
                        'Type': 'LONG',
                    })
                    in_trade = True
                    trade_entry = entry_price

        # Trailing stop for open trades: low of last 3-4 bars
        if in_trade and i >= 4:
            trailing_sl = min(df['Low'].iloc[i - 3:i + 1])
            # Check if trailing SL is hit
            if df['Low'].iloc[i] < trailing_sl and df['Low'].iloc[i] < trade_entry:
                exits.append({
                    'Date': df.index[i],
                    'Price': trailing_sl,
                    'Reason': 'Trailing SL (3-4 bar low)',
                })
                in_trade = False
            # Check if extension target hit
            elif df['High'].iloc[i] >= extensions.get('127.2%', swing_high * 1.1):
                exits.append({
                    'Date': df.index[i],
                    'Price': extensions['127.2%'],
                    'Reason': 'TP (Fib 127.2% Extension)',
                })
                in_trade = False

    backtest = _calc_backtest_stats(fib_entries, exits, 'LONG')

    fib_data = {
        'swing_high': swing_high,
        'swing_low': swing_low,
        'swing_high_idx': swing_high_idx,
        'swing_low_idx': swing_low_idx,
        'retracements': retracements,
        'extensions': extensions,
    }

    return fib_entries, exits, backtest, df, fib_data


# ═══════════════════════════════════════════════════════════
# STRATEGY 4: PRICE ACTION (INSIDE BAR BREAKOUT)
# ═══════════════════════════════════════════════════════════

def strategy_price_action(df: pd.DataFrame):
    """Price Action Breakout Strategy (Inside Bar).

    Rules:
    - Target: Inside Bar (Current High < Prev High AND Current Low > Prev Low)
    - Entry: Break of Inside Bar High (LONG) or Low (SHORT)
    - Stop-Loss: Opposite side of Inside Bar
    - Take Profit: 1.5x Risk
    
    Returns:
        short_signals, long_signals, backtest, df_s
    """
    ema_21 = calc_ema(df['Close'], 21)
    ema_55 = calc_ema(df['Close'], 55)

    df_s = df.copy()
    df_s['EMA_21'] = ema_21
    df_s['EMA_55'] = ema_55
    
    short_signals = []
    long_signals = []
    
    for i in range(2, len(df_s)):
        if pd.isna(df_s['EMA_55'].iloc[i]):
            continue
            
        # Inside bar detection
        prev_high = df_s['High'].iloc[i-1]
        prev_low = df_s['Low'].iloc[i-1]
        curr_high = df_s['High'].iloc[i]
        curr_low = df_s['Low'].iloc[i]
        
        is_inside_bar = (curr_high < prev_high) and (curr_low > prev_low)
        
        if is_inside_bar:
            # Check for breakout in the next candle (i+1) if it exists, or simulated
            if i + 1 < len(df_s):
                next_high = df_s['High'].iloc[i+1]
                next_low = df_s['Low'].iloc[i+1]
                
                # Uptrend context
                if ema_21.iloc[i] > ema_55.iloc[i]:
                    if next_high > curr_high: # Breakout long
                        entry_price = curr_high
                        sl_price = curr_low
                        risk = entry_price - sl_price
                        tp_price = entry_price + (1.5 * risk) if risk > 0 else entry_price * 1.02
                        
                        long_signals.append({
                            'Date': df_s.index[i+1],
                            'Entry': entry_price,
                            'SL': sl_price,
                            'TP': tp_price,
                            'Type': 'LONG',
                            'Pattern': 'Inside Bar Breakout UP'
                        })
                # Downtrend context
                elif ema_21.iloc[i] < ema_55.iloc[i]:
                    if next_low < curr_low: # Breakout short
                        entry_price = curr_low
                        sl_price = curr_high
                        risk = sl_price - entry_price
                        tp_price = entry_price - (1.5 * risk) if risk > 0 else entry_price * 0.98
                        
                        short_signals.append({
                            'Date': df_s.index[i+1],
                            'Entry': entry_price,
                            'SL': sl_price,
                            'TP': tp_price,
                            'Type': 'SHORT',
                            'Pattern': 'Inside Bar Breakout DOWN'
                        })

    backtest = _calc_backtest_stats_mixed(short_signals, long_signals, df_s)
    return short_signals, long_signals, backtest, df_s


# ═══════════════════════════════════════════════════════════
# STRATEGY 5: ADVANCED CANDLESTICK REVERSAL
# ═══════════════════════════════════════════════════════════

def strategy_candlestick_reversal(df: pd.DataFrame):
    """Advanced Candlestick Reversal Strategy.

    Rules:
    - Identifies strong reversal patterns (Engulfing, Hammer/Shooting Star)
    - Confirms with RSI (Oversold < 35 for LONG, Overbought > 65 for SHORT)
    - Entry: Next candle open
    - SL: Pattern High/Low
    - TP: 2x Risk or next moving average support/resistance
    """
    rsi = calc_rsi(df['Close'], 14)
    sma_50 = calc_sma(df['Close'], 50)

    df_s = df.copy()
    df_s['RSI'] = rsi
    df_s['SMA_50'] = sma_50
    df_s['Bullish_Eng'] = detect_bullish_engulfing(df)
    df_s['Bearish_Eng'] = detect_bearish_engulfing(df)
    df_s['Hammer'] = detect_hammer(df)
    df_s['Shooting_Star'] = detect_shooting_star(df)
    
    short_signals = []
    long_signals = []
    
    for i in range(1, len(df_s)-1):
        if pd.isna(df_s['RSI'].iloc[i]):
            continue
            
        is_bull_eng = df_s['Bullish_Eng'].iloc[i]
        is_hammer = df_s['Hammer'].iloc[i]
        
        is_bear_eng = df_s['Bearish_Eng'].iloc[i]
        is_star = df_s['Shooting_Star'].iloc[i]
        
        # LONG signals
        if (is_bull_eng or is_hammer) and df_s['RSI'].iloc[i] < 35:
            entry_price = df_s['Open'].iloc[i+1] # Enter next day open
            sl_price = df_s['Low'].iloc[i] # Stop below pattern low
            risk = entry_price - sl_price
            
            # If entry is somehow below SL (gap down), invalidate
            if risk > 0:
                tp_price = entry_price + (2.0 * risk)
                if df_s['SMA_50'].iloc[i] > entry_price + risk:
                    tp_price = min(tp_price, df_s['SMA_50'].iloc[i])
                
                pattern_name = 'Bullish Engulfing' if is_bull_eng else 'Hammer'
                
                long_signals.append({
                    'Date': df_s.index[i+1],
                    'Entry': entry_price,
                    'SL': sl_price,
                    'TP': tp_price,
                    'Type': 'LONG',
                    'Pattern': pattern_name,
                    'RSI': df_s['RSI'].iloc[i]
                })
        
        # SHORT signals
        if (is_bear_eng or is_star) and df_s['RSI'].iloc[i] > 65:
            entry_price = df_s['Open'].iloc[i+1]
            sl_price = df_s['High'].iloc[i]
            risk = sl_price - entry_price
            
            if risk > 0:
                tp_price = entry_price - (2.0 * risk)
                if df_s['SMA_50'].iloc[i] < entry_price - risk:
                    tp_price = max(tp_price, df_s['SMA_50'].iloc[i])
                    
                pattern_name = 'Bearish Engulfing' if is_bear_eng else 'Shooting Star'
                
                short_signals.append({
                    'Date': df_s.index[i+1],
                    'Entry': entry_price,
                    'SL': sl_price,
                    'TP': tp_price,
                    'Type': 'SHORT',
                    'Pattern': pattern_name,
                    'RSI': df_s['RSI'].iloc[i]
                })

    backtest = _calc_backtest_stats_mixed(short_signals, long_signals, df_s)
    return short_signals, long_signals, backtest, df_s


# ═══════════════════════════════════════════════════════════
# BACKTEST HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def _calc_backtest_stats(entries, exits, direction='LONG', fee_bps=5.0, slippage_bps=3.0):
    """Calculate backtest statistics for entry/exit pairs.

    fee_bps and slippage_bps are applied on entry and exit (round-trip).
    """
    if not entries:
        return {'total_signals': 0, 'total_exits': 0, 'win_rate': 0,
                'avg_rr': 0, 'net_pnl_pct': 0, 'winners': 0, 'losers': 0}

    paired = min(len(entries), len(exits))
    winners = 0
    losers = 0
    total_rr = 0
    total_pnl = 0

    for i in range(paired):
        entry = entries[i]['Entry']
        exit_price = exits[i]['Price']
        sl = entries[i]['SL']

        risk = abs(entry - sl)
        if risk == 0:
            continue

        if direction == 'LONG':
            pnl = exit_price - entry
        else:
            pnl = entry - exit_price

        # Round-trip transaction cost model: fees + slippage on both legs.
        total_cost_pct = (2 * (fee_bps + slippage_bps)) / 10000
        pnl_pct = (pnl / entry) - total_cost_pct
        pnl = pnl_pct * entry

        rr = pnl / risk if risk > 0 else 0
        total_rr += rr
        total_pnl += pnl_pct * 100

        if pnl > 0:
            winners += 1
        else:
            losers += 1

    total_trades = winners + losers
    return {
        'total_signals': len(entries),
        'total_exits': len(exits),
        'win_rate': (winners / total_trades * 100) if total_trades > 0 else 0,
        'avg_rr': (total_rr / total_trades) if total_trades > 0 else 0,
        'net_pnl_pct': total_pnl,
        'winners': winners,
        'losers': losers,
    }


def _calc_backtest_stats_mixed(shorts, longs, df_s, fee_bps=5.0, slippage_bps=3.0):
    """Calculate combined stats for mixed long/short signals."""
    total = len(shorts) + len(longs)
    if total == 0:
        return {'total_signals': 0, 'win_rate': 0, 'avg_rr': 0,
                'short_count': 0, 'long_count': 0}

    all_rr = []
    for sig in shorts + longs:
        risk = abs(sig['Entry'] - sig['SL'])
        reward = abs(sig['Entry'] - sig['TP'])
        rr_raw = reward / risk if risk > 0 else 0
        total_cost_pct = (2 * (fee_bps + slippage_bps)) / 10000
        rr_cost = total_cost_pct / (risk / sig['Entry']) if risk > 0 and sig['Entry'] > 0 else 0
        rr = rr_raw - rr_cost
        all_rr.append(rr)

    return {
        'total_signals': total,
        'short_count': len(shorts),
        'long_count': len(longs),
        'avg_rr': np.mean(all_rr) if all_rr else 0,
        'win_rate': None,  # Can't compute without exits, show as theoretical
    }
