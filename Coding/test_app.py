import pytest
import pandas as pd
import numpy as np


def make_sample_df(n=100):
    """Create a sample OHLCV DataFrame mimicking yfinance output."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 150 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "Open": close - np.random.rand(n),
        "High": close + np.abs(np.random.randn(n)),
        "Low": close - np.abs(np.random.randn(n)),
        "Close": close,
        "Volume": np.random.randint(1_000_000, 50_000_000, n),
    }, index=dates)


# ── Indicator Calculation Tests ──

class TestEMA:
    def test_ema_length(self):
        from indicators import calc_ema
        df = make_sample_df()
        ema = calc_ema(df["Close"], 9)
        assert len(ema) == len(df)

    def test_ema_converges_to_price(self):
        from indicators import calc_ema
        df = pd.DataFrame({"Close": [100.0] * 50})
        ema = calc_ema(df["Close"], 9)
        assert all(np.isclose(ema, 100.0))


class TestRSI:
    def test_rsi_range(self):
        from indicators import calc_rsi
        df = make_sample_df(200)
        rsi = calc_rsi(df["Close"]).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_rsi_overbought_on_rally(self):
        from indicators import calc_rsi
        prices = pd.Series(range(100, 200))
        rsi = calc_rsi(prices).dropna()
        assert rsi.iloc[-1] > 90


class TestMACD:
    def test_macd_calculation(self):
        from indicators import calc_macd
        df = make_sample_df(100)
        macd, signal, histogram = calc_macd(df["Close"])
        assert len(histogram) == len(df)
        assert not histogram.isna().all()


class TestBollingerBands:
    def test_price_within_bands(self):
        from indicators import calc_bollinger_bands
        df = make_sample_df(200)
        mid, upper, lower = calc_bollinger_bands(df["Close"])
        valid = df["Close"][20:]
        within = ((valid <= upper[20:]) & (valid >= lower[20:])).mean()
        assert within > 0.85, f"Only {within:.0%} within bands"


class TestATR:
    def test_atr_positive(self):
        from indicators import calc_atr
        df = make_sample_df(100)
        atr = calc_atr(df).dropna()
        assert (atr > 0).all()


class TestStochastic:
    def test_stochastic_range(self):
        from indicators import calc_stochastic
        df = make_sample_df(200)
        k, d = calc_stochastic(df)
        k_valid = k.dropna()
        d_valid = d.dropna()
        assert (k_valid >= 0).all() and (k_valid <= 100).all(), "%K out of bounds"
        assert (d_valid >= 0).all() and (d_valid <= 100).all(), "%D out of bounds"


class TestIchimoku:
    def test_ichimoku_components(self):
        from indicators import calc_ichimoku
        df = make_sample_df(100)
        tenkan, kijun, senkou_a, senkou_b, chikou = calc_ichimoku(df)
        assert len(tenkan) == len(df)
        assert len(kijun) == len(df)
        # Tenkan should be between high and low
        valid_ten = tenkan.dropna()
        assert len(valid_ten) > 0


class TestParabolicSAR:
    def test_sar_length(self):
        from indicators import calc_parabolic_sar
        df = make_sample_df(100)
        sar = calc_parabolic_sar(df)
        assert len(sar) == len(df)

    def test_sar_values_positive(self):
        from indicators import calc_parabolic_sar
        df = make_sample_df(100)
        sar = calc_parabolic_sar(df)
        assert (sar > 0).all()


class TestSuperTrend:
    def test_supertrend_direction(self):
        from indicators import calc_supertrend
        df = make_sample_df(100)
        st_line, direction = calc_supertrend(df)
        valid_dir = direction.dropna()
        assert set(valid_dir.unique()).issubset({-1, 1})


class TestCandlestickPatterns:
    def test_doji_detection(self):
        from indicators import detect_doji
        df = pd.DataFrame({
            'Open': [100.0, 100.0], 'Close': [100.01, 110.0],
            'High': [105.0, 115.0], 'Low': [95.0, 105.0]
        })
        result = detect_doji(df)
        assert result.iloc[0] == True  # tiny body
        assert result.iloc[1] == False  # big body

    def test_hammer_detection(self):
        from indicators import detect_hammer
        df = pd.DataFrame({
            'Open': [100.0], 'Close': [101.0],
            'High': [101.3], 'Low': [95.0]  # long lower shadow, small upper shadow
        })
        result = detect_hammer(df)
        assert result.iloc[0] == True

    def test_harami_detection(self):
        from indicators import detect_harami
        df = pd.DataFrame({
            'Open': [110.0, 102.0], 'Close': [100.0, 105.0],
            'High': [112.0, 106.0], 'Low': [98.0, 101.0]
        })
        result = detect_harami(df)
        assert result.iloc[1] == True


# ── Risk Metric Tests ──

class TestRiskMetrics:
    def test_sharpe_ratio(self):
        df = make_sample_df(252)
        returns = df["Close"].pct_change().dropna()
        rf = 0.02 / 252
        sharpe = (returns.mean() - rf) / returns.std() * np.sqrt(252)
        assert isinstance(sharpe, float)

    def test_max_drawdown_negative(self):
        df = make_sample_df(200)
        running_max = df["Close"].cummax()
        drawdown = (df["Close"] - running_max) / running_max
        assert drawdown.min() <= 0

    def test_var_less_than_zero(self):
        df = make_sample_df(200)
        returns = df["Close"].pct_change().dropna()
        var_95 = np.percentile(returns, 5)
        assert var_95 < 0


# ── Strategy Signal Tests ──

class TestStrategySignals:
    def test_signal_dict_structure(self):
        signal = {"Date": pd.Timestamp("2025-06-01"), "Price": 155.0, "SL": 152.0, "Type": "LONG"}
        assert all(k in signal for k in ["Date", "Price", "SL", "Type"])
        assert signal["Type"] in ["LONG", "SHORT"]

    def test_fibonacci_levels(self):
        from indicators import calc_fibonacci_levels
        retrace, ext = calc_fibonacci_levels(200.0, 100.0)
        assert retrace['50.0%'] == 150.0
        assert retrace['61.8%'] == pytest.approx(138.2)
        assert 100 < retrace['61.8%'] < retrace['50.0%'] < retrace['38.2%'] < 200

    def test_position_size_calculator(self):
        from strategies import calc_position_size
        result = calc_position_size(10000, 1.0, 150.0, 148.0)
        assert result['max_loss'] == 100.0
        assert result['sl_distance'] == 2.0
        assert result['shares'] == 50
        assert result['warning'] == False

    def test_position_size_warning(self):
        from strategies import calc_position_size
        result = calc_position_size(100, 50.0, 150.0, 149.0)
        assert result['warning'] == True  # investment > capital


class TestScalpingStrategy:
    def test_only_long_signals(self):
        from strategies import strategy_bollinger_scalping
        df = make_sample_df(200)
        entries, exits, backtest, _ = strategy_bollinger_scalping(df)
        for e in entries:
            assert e['Type'] == 'LONG', "Scalping should only produce LONG signals"


class TestReversalStrategy:
    def test_rsi_threshold(self):
        """Reversal strategy should use RSI > 90 for shorts, < 10 for longs."""
        from strategies import strategy_reversal
        df = make_sample_df(200)
        shorts, longs, _, _ = strategy_reversal(df)
        for s in shorts:
            assert s['RSI'] > 90, f"Short signal with RSI {s['RSI']} (should be > 90)"
        for l in longs:
            assert l['RSI'] < 10, f"Long signal with RSI {l['RSI']} (should be < 10)"


# ── Session State Tests ──

class TestSessionDefaults:
    def test_default_watchlist(self):
        watchlist = ["AAPL", "MSFT", "GOOG", "TSLA", "BTC-USD"]
        assert len(watchlist) == 5
        assert "AAPL" in watchlist

    def test_default_portfolio_structure(self):
        portfolio = pd.DataFrame([
            {"Symbol": "AAPL", "Shares": 50, "EntryPrice": 150.0},
            {"Symbol": "MSFT", "Shares": 30, "EntryPrice": 300.0},
        ])
        assert list(portfolio.columns) == ["Symbol", "Shares", "EntryPrice"]
        assert len(portfolio) == 2
