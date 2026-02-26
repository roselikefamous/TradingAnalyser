import pandas as pd

from trading_terminal.backtest_engine import BacktestConfig, run_backtest
from trading_terminal.contracts import BacktestSignal
from trading_terminal.risk_engine import RiskConfig


def _ohlc_for_tp_hit():
    idx = pd.date_range("2025-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0, 100.0],
            "High": [100.0, 102.0, 101.0, 100.0],
            "Low": [99.5, 99.5, 99.0, 99.0],
            "Close": [100.0, 101.5, 100.5, 100.0],
        },
        index=idx,
    )


def test_backtest_executes_tp_trade():
    df = _ohlc_for_tp_hit()
    sig = BacktestSignal(
        date=df.index[0],
        side="LONG",
        sl=99.0,
        tp=101.5,
        quantity=10,
        symbol="TEST",
    )
    res = run_backtest(df, [sig], BacktestConfig(initial_cash=10_000, fee_bps=0.0, slippage_bps=0.0))
    assert len(res.trades) == 1
    assert res.trades[0].exit_reason == "TP"
    assert res.trades[0].net_pnl > 0


def test_backtest_short_tp_trade_is_profitable():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [100.5, 100.2, 100.1],
            "Low": [99.8, 97.8, 97.5],
            "Close": [100.0, 98.2, 98.0],
        },
        index=idx,
    )
    sig = BacktestSignal(date=idx[0], side="SHORT", sl=101.0, tp=98.0, quantity=10, symbol="TEST")
    res = run_backtest(df, [sig], BacktestConfig(initial_cash=10_000, fee_bps=0.0, slippage_bps=0.0))
    assert len(res.trades) == 1
    assert res.trades[0].side == "SHORT"
    assert res.trades[0].exit_reason == "TP"
    assert res.trades[0].net_pnl > 0


def test_backtest_conservative_when_tp_and_sl_same_bar():
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0],
            "High": [100.0, 103.0],
            "Low": [99.0, 97.0],
            "Close": [100.0, 100.0],
        },
        index=idx,
    )
    sig = BacktestSignal(date=idx[0], side="LONG", sl=98.0, tp=102.0, quantity=10, symbol="TEST")
    res = run_backtest(df, [sig], BacktestConfig(initial_cash=10_000, fee_bps=0.0, slippage_bps=0.0))
    assert len(res.trades) == 1
    assert "conservative SL" in res.trades[0].exit_reason
    assert res.trades[0].net_pnl < 0


def test_risk_engine_halts_on_daily_loss():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [101.0, 100.5, 100.2],
            "Low": [99.0, 97.5, 99.5],
            "Close": [100.0, 98.5, 100.0],
        },
        index=idx,
    )
    # First signal intentionally loses big; second should be blocked by risk halt.
    sig1 = BacktestSignal(date=df.index[0], side="LONG", sl=98.0, tp=110.0, quantity=500, symbol="TEST")
    sig2 = BacktestSignal(date=df.index[2], side="LONG", sl=99.0, tp=101.0, quantity=10, symbol="TEST")
    risk_cfg = RiskConfig(max_daily_loss_pct=0.0004, max_drawdown_pct=1.0, max_open_positions=1, max_gross_exposure_pct=1.0)
    res = run_backtest(df, [sig1, sig2], BacktestConfig(initial_cash=100_000, fee_bps=0.0, slippage_bps=0.0), risk_cfg)
    # At least one trade exists, and halt eventually becomes true.
    assert len(res.trades) >= 1
    assert any(s.halted for s in res.risk_snapshots)


def test_intrabar_mode_changes_same_bar_outcome():
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0],
            "High": [100.0, 103.0],
            "Low": [99.0, 97.0],
            "Close": [100.0, 100.0],
        },
        index=idx,
    )
    sig = BacktestSignal(date=idx[0], side="LONG", sl=98.0, tp=102.0, quantity=10, symbol="TEST")

    cons = run_backtest(df, [sig], BacktestConfig(initial_cash=10_000, fee_bps=0.0, slippage_bps=0.0, intrabar_mode="conservative"))
    opt = run_backtest(df, [sig], BacktestConfig(initial_cash=10_000, fee_bps=0.0, slippage_bps=0.0, intrabar_mode="optimistic"))
    mid = run_backtest(df, [sig], BacktestConfig(initial_cash=10_000, fee_bps=0.0, slippage_bps=0.0, intrabar_mode="mid"))

    assert cons.trades[0].exit_reason.endswith("conservative SL)")
    assert opt.trades[0].exit_reason.endswith("optimistic TP)")
    assert mid.trades[0].exit_reason.endswith("midpoint)")
    assert cons.trades[0].net_pnl < mid.trades[0].net_pnl < opt.trades[0].net_pnl
