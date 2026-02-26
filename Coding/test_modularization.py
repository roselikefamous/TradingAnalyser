import numpy as np
import pandas as pd

from indicators import calc_atr, calc_bollinger_bands, calc_ema
from trading_terminal.contracts import PositionSizeResult, StrategyExit, StrategySignal
from trading_terminal.strategies import (
    compute_position_size,
    run_bollinger_scalping,
    to_legacy_exit_dicts,
    to_legacy_signal_dicts,
)


def _sample_ohlcv(n=120):
    np.random.seed(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame(
        {
            "Open": close - 0.4,
            "High": close + 0.8,
            "Low": close - 0.8,
            "Close": close,
            "Volume": np.random.randint(100_000, 2_000_000, n),
        },
        index=idx,
    )


def test_ema_known_values_span_2():
    series = pd.Series([1.0, 2.0, 3.0])
    ema = calc_ema(series, span=2)
    expected = np.array([1.0, 1.6666666667, 2.5555555556])
    assert np.allclose(ema.values, expected, atol=1e-6)


def test_atr_period_one_matches_true_range():
    df = pd.DataFrame(
        {
            "High": [10.0, 12.0, 11.0],
            "Low": [8.0, 9.0, 10.0],
            "Close": [9.0, 11.0, 10.5],
        }
    )
    atr = calc_atr(df, period=1)
    assert np.allclose(atr.values, np.array([2.0, 3.0, 1.0]), atol=1e-9)


def test_bollinger_constant_series_collapse_to_midline():
    series = pd.Series([50.0] * 30)
    mid, upper, lower = calc_bollinger_bands(series, period=20, std_dev=2)
    valid = slice(19, None)
    assert np.allclose(mid.iloc[valid], 50.0, atol=1e-9)
    assert np.allclose(upper.iloc[valid], 50.0, atol=1e-9)
    assert np.allclose(lower.iloc[valid], 50.0, atol=1e-9)


def test_strategy_signal_contract_roundtrip():
    payload = {
        "Date": pd.Timestamp("2025-06-01"),
        "Entry": 155.0,
        "SL": 152.0,
        "TP": 160.0,
        "Type": "LONG",
        "Pattern": "Hammer",
        "RSI": 32.1,
        "Level": "61.8%",
        "StochConfirm": True,
    }
    signal = StrategySignal.from_legacy(payload)
    roundtrip = signal.to_legacy()
    assert roundtrip["Type"] == "LONG"
    assert roundtrip["Entry"] == 155.0
    assert roundtrip["SL"] == 152.0
    assert roundtrip["TP"] == 160.0
    assert roundtrip["Pattern"] == "Hammer"
    assert roundtrip["StochConfirm"] is True


def test_strategy_exit_contract_roundtrip():
    payload = {"Date": pd.Timestamp("2025-06-01"), "Price": 151.5, "Reason": "SL Hit"}
    exit_contract = StrategyExit.from_legacy(payload)
    assert exit_contract.to_legacy() == payload


def test_position_size_contract_mapping():
    result = compute_position_size(10_000, 1.0, 150.0, 148.0)
    assert isinstance(result, PositionSizeResult)
    assert result.max_loss == 100.0
    assert result.sl_distance == 2.0
    assert result.shares == 50
    assert result.warning is False


def test_bollinger_service_returns_contracts_and_legacy_conversion():
    df = _sample_ohlcv(220)
    entries, exits, _, _ = run_bollinger_scalping(df)
    assert all(isinstance(s, StrategySignal) for s in entries)
    assert all(isinstance(e, StrategyExit) for e in exits)

    legacy_entries = to_legacy_signal_dicts(entries)
    legacy_exits = to_legacy_exit_dicts(exits)
    for entry in legacy_entries:
        assert {"Date", "Entry", "SL", "Type"}.issubset(entry.keys())
    for exit_row in legacy_exits:
        assert {"Date", "Price", "Reason"}.issubset(exit_row.keys())

