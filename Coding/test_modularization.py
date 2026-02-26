import numpy as np
import pandas as pd

from indicators import calc_atr, calc_bollinger_bands, calc_ema
from indicators import calc_rsi, calc_stochastic
from trading_terminal.contracts import PositionSizeResult, StrategyExit, StrategySignal
from trading_terminal.strategies import (
    compute_position_size,
    run_bollinger_scalping,
    run_walk_forward_backtest,
    to_legacy_exit_dicts,
    to_legacy_signal_dicts,
)
from strategies import _calc_backtest_stats


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


def test_rsi_flat_series_converges_to_50():
    prices = pd.Series([100.0] * 40)
    rsi = calc_rsi(prices, period=14).dropna()
    assert len(rsi) > 0
    assert np.allclose(rsi.values, 50.0, atol=1e-9)


def test_stochastic_constant_range_stays_bounded():
    df = pd.DataFrame(
        {
            "High": [100.0] * 40,
            "Low": [100.0] * 40,
            "Close": [100.0] * 40,
        }
    )
    k, d = calc_stochastic(df)
    assert np.isfinite(k.dropna()).all()
    assert np.isfinite(d.dropna()).all()
    assert ((k.dropna() >= 0) & (k.dropna() <= 100)).all()
    assert ((d.dropna() >= 0) & (d.dropna() <= 100)).all()


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


def test_backtest_cost_model_reduces_nominal_edge():
    entries = [{"Entry": 100.0, "SL": 99.0}]
    exits = [{"Price": 101.0}]
    no_cost = _calc_backtest_stats(entries, exits, fee_bps=0.0, slippage_bps=0.0)
    with_cost = _calc_backtest_stats(entries, exits, fee_bps=10.0, slippage_bps=10.0)
    assert with_cost["net_pnl_pct"] < no_cost["net_pnl_pct"]


def test_walk_forward_input_validation():
    df = _sample_ohlcv(40)
    try:
        run_walk_forward_backtest(df, train_size=50, test_size=10, step_size=10)
        assert False, "Expected ValueError for insufficient rows"
    except ValueError:
        assert True


def test_walk_forward_runs_with_monkeypatched_strategy(monkeypatch):
    df = _sample_ohlcv(90)

    def fake_runner(window_df):
        entries = [{"Date": window_df.index[0], "Entry": 100.0, "SL": 99.0, "Type": "LONG"}]
        exits = [{"Date": window_df.index[-1], "Price": 101.0, "Reason": "TP"}]
        backtest = {"win_rate": 100.0, "avg_rr": 1.0, "net_pnl_pct": 1.0}
        return entries, exits, backtest, window_df

    monkeypatch.setattr("trading_terminal.strategies.service.strategy_bollinger_scalping", fake_runner)
    folds, summary = run_walk_forward_backtest(
        df,
        strategy_name="bollinger_scalping",
        train_size=40,
        test_size=20,
        step_size=10,
    )
    assert len(folds) > 0
    assert summary.folds == len(folds)
    assert summary.profitable_folds == len(folds)


def test_walk_forward_mixed_strategy_with_simulated_exits(monkeypatch):
    df = _sample_ohlcv(120)

    def fake_mixed_runner(window_df):
        entry_date = window_df.index[min(5, len(window_df) - 1)]
        shorts = []
        longs = [
            {
                "Date": entry_date,
                "Entry": 100.0,
                "SL": 99.0,
                "TP": 101.0,
                "Type": "LONG",
                "Pattern": "Test",
            }
        ]
        backtest = {"win_rate": 0.0, "avg_rr": 0.0, "net_pnl_pct": 0.0}
        return shorts, longs, backtest, window_df

    monkeypatch.setattr("trading_terminal.strategies.service.strategy_reversal", fake_mixed_runner)
    folds, summary = run_walk_forward_backtest(
        df,
        strategy_name="reversal",
        train_size=60,
        test_size=20,
        step_size=20,
    )
    assert len(folds) > 0
    assert all(f.test_exits >= 0 for f in folds)
    assert summary.folds == len(folds)
