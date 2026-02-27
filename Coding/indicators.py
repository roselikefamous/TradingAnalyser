"""
indicators.py – Complete Technical Indicator Engine
All indicator calculations for the Trading Terminal.
"""
import numpy as np
import pandas as pd
from numba import njit


# ═══════════════════════════════════════════════════════════
# 1. TREND INDICATORS
# ═══════════════════════════════════════════════════════════

def calc_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calc_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window).mean()


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price (intraday reset style)."""
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical * df['Volume']).cumsum() / df['Volume'].cumsum()


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    """MACD line, Signal line, Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_ichimoku(df: pd.DataFrame, tenkan=9, kijun=26, senkou_b=52):
    """Ichimoku Cloud components."""
    high = df['High']
    low = df['Low']
    close = df['Close']

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_line = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    chikou = close.shift(-kijun)

    return tenkan_sen, kijun_sen, senkou_a, senkou_b_line, chikou


@njit
def _parabolic_sar_numba(high, low, af_start, af_step, af_max):
    n = len(high)
    sar = np.zeros(n)
    af = af_start
    uptrend = True
    ep = low[0]
    sar[0] = high[0]

    for i in range(1, n):
        if uptrend:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = min(sar[i], low[i - 1])
            if i >= 2:
                sar[i] = min(sar[i], low[i - 2])

            if low[i] < sar[i]:
                uptrend = False
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = max(sar[i], high[i - 1])
            if i >= 2:
                sar[i] = max(sar[i], high[i - 2])

            if high[i] > sar[i]:
                uptrend = True
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
    return sar


def calc_parabolic_sar(df: pd.DataFrame, af_start=0.02, af_step=0.02, af_max=0.2):
    """Parabolic SAR (Wilder's method) accelerated with Numba."""
    high = df['High'].values
    low = df['Low'].values
    sar = _parabolic_sar_numba(high, low, af_start, af_step, af_max)
    return pd.Series(sar, index=df.index, name='SAR')


@njit
def _supertrend_numba(close, upper_band, lower_band, period):
    n = len(close)
    supertrend = np.full(n, np.nan)
    direction = np.ones(n) # 1 = up, -1 = down
    
    supertrend[period] = lower_band[period]
    direction[period] = 1
    
    for i in range(period + 1, n):
        if direction[i - 1] == 1:
            # Was in uptrend
            prev_st = supertrend[i - 1]
            supertrend[i] = max(lower_band[i], prev_st)
            if close[i] < supertrend[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
        else:
            # Was in downtrend
            prev_st = supertrend[i - 1]
            supertrend[i] = min(upper_band[i], prev_st)
            if close[i] > supertrend[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
    return supertrend, direction


def calc_supertrend(df: pd.DataFrame, period=10, multiplier=3.0):
    """SuperTrend indicator accelerated with Numba."""
    atr = calc_atr(df, period)
    hl2 = (df['High'] + df['Low']) / 2
    upper_band = (hl2 + multiplier * atr).values
    lower_band = (hl2 - multiplier * atr).values
    close = df['Close'].values
    
    st_values, dir_values = _supertrend_numba(close, upper_band, lower_band, period)
    
    return pd.Series(st_values, index=df.index, name='SuperTrend'), \
           pd.Series(dir_values, index=df.index, name='Direction')


# ═══════════════════════════════════════════════════════════
# 2. MOMENTUM & OSCILLATOR INDICATORS
# ═══════════════════════════════════════════════════════════

def calc_rsi(series: pd.Series, period=14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    loss_safe = loss.replace(0, np.nan)
    rs = gain / loss_safe
    rsi = 100 - (100 / (1 + rs))

    # Define edge cases explicitly: no losses => RSI 100, no gains => RSI 0, flat => 50.
    rsi = rsi.where(~((loss == 0) & (gain > 0)), 100)
    rsi = rsi.where(~((gain == 0) & (loss > 0)), 0)
    rsi = rsi.where(~((gain == 0) & (loss == 0)), 50)
    return rsi.clip(lower=0, upper=100)


def calc_stochastic(df: pd.DataFrame, k_period=14, d_period=3, smooth_k=3):
    """Stochastic Oscillator – %K and %D."""
    lowest_low = df['Low'].rolling(k_period).min()
    highest_high = df['High'].rolling(k_period).max()
    range_span = (highest_high - lowest_low).replace(0, np.nan)
    raw_k = 100 * (df['Close'] - lowest_low) / range_span
    k = raw_k.rolling(smooth_k).mean()  # smoothed %K
    d = k.rolling(d_period).mean()  # %D
    return k.clip(lower=0, upper=100), d.clip(lower=0, upper=100)


# ═══════════════════════════════════════════════════════════
# 3. VOLATILITY INDICATORS
# ═══════════════════════════════════════════════════════════

def calc_bollinger_bands(series: pd.Series, period=20, std_dev=2):
    """Bollinger Bands – Mid, Upper, Lower."""
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return mid, upper, lower


def calc_atr(df: pd.DataFrame, period=14) -> pd.Series:
    """Average True Range."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ═══════════════════════════════════════════════════════════
# 4. VOLUME INDICATORS
# ═══════════════════════════════════════════════════════════

def calc_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()


@njit
def _volume_profile_numba(highs, lows, volumes_in, price_bins, bins, price_range):
    volumes_out = np.zeros(bins)
    bin_size = price_range / bins
    
    for i in range(len(highs)):
        h = highs[i]
        l = lows[i]
        v = volumes_in[i]
        
        # Calculate weight distributed across overlapping bins
        span = h - l
        weight = v / max(1.0, span / bin_size) if span > 0 else v
        
        for j in range(bins):
            if l <= price_bins[j + 1] and h >= price_bins[j]:
                volumes_out[j] += weight
                
    return volumes_out


def calc_volume_profile(df: pd.DataFrame, bins=30):
    """Volume Profile accelerated with Numba."""
    price_min = df['Low'].min()
    price_max = df['High'].max()
    price_bins = np.linspace(price_min, price_max, bins + 1)
    price_range = price_max - price_min
    
    if price_range == 0:
        return (price_bins[:-1] + price_bins[1:]) / 2, np.zeros(bins)

    volumes = _volume_profile_numba(
        df['High'].values, 
        df['Low'].values, 
        df['Volume'].values, 
        price_bins, 
        bins, 
        price_range
    )

    price_levels = (price_bins[:-1] + price_bins[1:]) / 2
    return price_levels, volumes


# ═══════════════════════════════════════════════════════════
# 5. MARKET STRUCTURE
# ═══════════════════════════════════════════════════════════

def calc_adx(df: pd.DataFrame, period=14):
    """Average Directional Index with +DI and -DI."""
    atr = calc_atr(df, period)
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


def calc_pivot_points(df: pd.DataFrame):
    """Classic Pivot Points – PP, R1, S1."""
    pp = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    r1 = (2 * pp) - df['Low'].shift(1)
    s1 = (2 * pp) - df['High'].shift(1)
    return pp, r1, s1


def calc_fibonacci_levels(swing_high, swing_low):
    """Fibonacci Retracement and Extension levels."""
    diff = swing_high - swing_low
    retracements = {
        '0.0%': swing_high,
        '23.6%': swing_high - diff * 0.236,
        '38.2%': swing_high - diff * 0.382,
        '50.0%': swing_high - diff * 0.500,
        '61.8%': swing_high - diff * 0.618,
        '78.6%': swing_high - diff * 0.786,
        '100.0%': swing_low,
    }
    extensions = {
        '127.2%': swing_high + diff * 0.272,
        '161.8%': swing_high + diff * 0.618,
    }
    return retracements, extensions


def find_swing_points(df: pd.DataFrame, window=5):
    """Find swing highs and swing lows (local pivots)."""
    swing_highs = []
    swing_lows = []
    swing_high_idx = []
    swing_low_idx = []

    for i in range(window, len(df) - window):
        if df['High'].iloc[i] == df['High'].iloc[i - window:i + window + 1].max():
            swing_highs.append(df['High'].iloc[i])
            swing_high_idx.append(df.index[i])
        if df['Low'].iloc[i] == df['Low'].iloc[i - window:i + window + 1].min():
            swing_lows.append(df['Low'].iloc[i])
            swing_low_idx.append(df.index[i])

    return swing_highs, swing_lows, swing_high_idx, swing_low_idx


def auto_fibonacci(df: pd.DataFrame, lookback=60):
    """Auto-detect major swing and compute Fibonacci levels."""
    window = min(lookback, len(df) - 1)
    recent = df.iloc[-window:]
    swing_low = recent['Low'].min()
    swing_high = recent['High'].max()
    swing_low_idx = recent['Low'].idxmin()
    swing_high_idx = recent['High'].idxmax()
    retracements, extensions = calc_fibonacci_levels(swing_high, swing_low)
    return retracements, extensions, swing_high, swing_low, swing_high_idx, swing_low_idx


# ═══════════════════════════════════════════════════════════
# 6. CANDLESTICK PATTERN RECOGNITION
# ═══════════════════════════════════════════════════════════

def detect_doji(df: pd.DataFrame, threshold=0.1) -> pd.Series:
    """Doji: tiny body relative to range."""
    body = np.abs(df['Close'] - df['Open'])
    total = df['High'] - df['Low']
    return body <= total * threshold


def detect_spinning_top(df: pd.DataFrame, body_pct=0.3, shadow_ratio=1.5) -> pd.Series:
    """Spinning Top: small body with long upper AND lower shadows."""
    body = np.abs(df['Close'] - df['Open'])
    total = df['High'] - df['Low']
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    return (body <= total * body_pct) & (upper_shadow >= body * shadow_ratio) & (lower_shadow >= body * shadow_ratio)


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """Hammer: small body at top, long lower shadow (2-3x body)."""
    body = np.abs(df['Close'] - df['Open'])
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    return (lower_shadow >= 2 * body) & (upper_shadow < body * 0.5) & (body > 0)


def detect_hanging_man(df: pd.DataFrame) -> pd.Series:
    """Hanging Man: same shape as hammer but at top of uptrend."""
    return detect_hammer(df)


def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
    """Shooting Star: small body at bottom, long upper shadow."""
    body = np.abs(df['Close'] - df['Open'])
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    return (upper_shadow >= 2 * body) & (lower_shadow < body * 0.5) & (body > 0)


def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bullish Engulfing: current bullish candle engulfs previous bearish candle."""
    c_open, c_close = df['Open'], df['Close']
    p_open, p_close = df['Open'].shift(1), df['Close'].shift(1)
    
    pattern = (c_close > c_open) & (p_close < p_open) & (c_close > p_open) & (c_open < p_close)
    return pattern.fillna(False)


def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bearish Engulfing: current bearish candle engulfs previous bullish candle."""
    c_open, c_close = df['Open'], df['Close']
    p_open, p_close = df['Open'].shift(1), df['Close'].shift(1)
    
    pattern = (c_close < c_open) & (p_close > p_open) & (c_open > p_close) & (c_close < p_open)
    return pattern.fillna(False)


def detect_harami(df: pd.DataFrame) -> pd.Series:
    """Bullish Harami: small body within previous large bearish body."""
    c_open, c_close = df['Open'], df['Close']
    p_open, p_close = df['Open'].shift(1), df['Close'].shift(1)
    c_body = (c_close - c_open).abs()
    p_body = (p_close - p_open).abs()
    
    pattern = (p_close < p_open) & (c_close > c_open) & \
              (c_close.combine(c_open, max) < p_open) & \
              (c_close.combine(c_open, min) > p_close) & \
              (c_body < p_body * 0.5)
              
    return pattern.fillna(False)


# ═══════════════════════════════════════════════════════════
# 7. HEIKIN-ASHI
# ═══════════════════════════════════════════════════════════

@njit
def _heikin_ashi_numba(opens, highs, lows, closes):
    n = len(opens)
    ha_open = np.zeros(n)
    ha_close = (opens + highs + lows + closes) / 4
    
    ha_open[0] = (opens[0] + closes[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
        
    ha_high = np.maximum(np.maximum(ha_open, ha_close), highs)
    ha_low = np.minimum(np.minimum(ha_open, ha_close), lows)
    
    return ha_open, ha_high, ha_low, ha_close


def calc_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Heikin-Ashi candles accelerated with Numba."""
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    ha_open, ha_high, ha_low, ha_close = _heikin_ashi_numba(opens, highs, lows, closes)
    
    ha = pd.DataFrame({
        'Open': ha_open,
        'High': ha_high,
        'Low': ha_low,
        'Close': ha_close
    }, index=df.index)
    return ha


# ═══════════════════════════════════════════════════════════
# 8. COMPOSITE: Apply all indicators to a DataFrame
# ═══════════════════════════════════════════════════════════

def apply_core_indicators(df: pd.DataFrame, ema1_len=9, ema2_len=21, rsi_len=14):
    """Apply all core indicators to the DataFrame in-place and return it."""
    df = df.copy()

    # EMAs
    df['EMA_1'] = calc_ema(df['Close'], ema1_len)
    df['EMA_2'] = calc_ema(df['Close'], ema2_len)
    df['EMA_55'] = calc_ema(df['Close'], 55)
    df['EMA_10'] = calc_ema(df['Close'], 10)
    df['EMA_21'] = calc_ema(df['Close'], 21)
    df['SMA_50'] = calc_sma(df['Close'], 50)
    df['SMA_200'] = calc_sma(df['Close'], 200)

    # Daily Return
    df['Daily_Return'] = df['Close'].pct_change()

    # VWAP & OBV
    if 'Volume' in df.columns:
        df['VWAP'] = calc_vwap(df)
        df['OBV'] = calc_obv(df)

    # RSI
    df['RSI'] = calc_rsi(df['Close'], rsi_len)

    # MACD
    df['MACD'], df['Signal'], df['MACD_Hist'] = calc_macd(df['Close'])

    # ATR
    df['ATR'] = calc_atr(df)

    # Bollinger Bands
    df['BB_Mid'], df['BB_Upper'], df['BB_Lower'] = calc_bollinger_bands(df['Close'])

    # Stochastic
    df['Stoch_K'], df['Stoch_D'] = calc_stochastic(df)

    # ADX
    df['ADX'], df['Plus_DI'], df['Minus_DI'] = calc_adx(df)

    # Candlestick Patterns
    df['Doji'] = detect_doji(df)
    df['Spinning_Top'] = detect_spinning_top(df)
    df['Hammer'] = detect_hammer(df)
    df['Shooting_Star'] = detect_shooting_star(df)
    df['Bullish_Engulfing'] = detect_bullish_engulfing(df)
    df['Bearish_Engulfing'] = detect_bearish_engulfing(df)
    df['Harami'] = detect_harami(df)

    return df
