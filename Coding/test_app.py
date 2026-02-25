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
        df = make_sample_df()
        ema = df["Close"].ewm(span=9, adjust=False).mean()
        assert len(ema) == len(df)

    def test_ema_converges_to_price(self):
        """EMA of a constant series should equal the constant."""
        df = pd.DataFrame({"Close": [100.0] * 50})
        ema = df["Close"].ewm(span=9, adjust=False).mean()
        assert all(np.isclose(ema, 100.0))


class TestRSI:
    def _calc_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def test_rsi_range(self):
        df = make_sample_df(200)
        rsi = self._calc_rsi(df["Close"]).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_rsi_overbought_on_rally(self):
        """A straight-up rally should produce RSI near 100."""
        prices = pd.Series(range(100, 200))
        rsi = self._calc_rsi(prices).dropna()
        assert rsi.iloc[-1] > 90


class TestMACD:
    def test_macd_calculation(self):
        df = make_sample_df(100)
        macd = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        assert len(histogram) == len(df)
        assert not histogram.isna().all()


class TestBollingerBands:
    def test_price_within_bands(self):
        """Most prices should fall within 2-std Bollinger Bands."""
        df = make_sample_df(200)
        mid = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        valid = df["Close"][20:]  # skip NaN warmup
        within = ((valid <= upper[20:]) & (valid >= lower[20:])).mean()
        assert within > 0.85, f"Only {within:.0%} within bands"


class TestATR:
    def test_atr_positive(self):
        df = make_sample_df(100)
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        atr = ranges.max(axis=1).rolling(14).mean().dropna()
        assert (atr > 0).all()


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
        assert var_95 < 0, "95% VaR should be negative"


# ── Strategy Signal Tests ──

class TestStrategySignals:
    def test_signal_dict_structure(self):
        """Verify expected signal dictionary keys."""
        signal = {"Date": pd.Timestamp("2025-06-01"), "Price": 155.0, "SL": 152.0, "Type": "LONG"}
        assert all(k in signal for k in ["Date", "Price", "SL", "Type"])
        assert signal["Type"] in ["LONG", "SHORT"]

    def test_fibonacci_levels(self):
        """Fibonacci retracements should be between swing low and high."""
        swing_low, swing_high = 100.0, 200.0
        diff = swing_high - swing_low
        fib_382 = swing_high - diff * 0.382
        fib_500 = swing_high - diff * 0.500
        fib_618 = swing_high - diff * 0.618
        assert swing_low < fib_618 < fib_500 < fib_382 < swing_high


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
