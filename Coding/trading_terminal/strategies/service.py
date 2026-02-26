import pandas as pd

from strategies import (
    calc_position_size,
    strategy_bollinger_scalping,
    strategy_fibonacci_swing,
    strategy_reversal,
)
from trading_terminal.contracts import PositionSizeResult, StrategyExit, StrategySignal


def to_legacy_signal_dicts(signals: list[StrategySignal]) -> list[dict]:
    return [s.to_legacy() for s in signals]


def to_legacy_exit_dicts(exits: list[StrategyExit]) -> list[dict]:
    return [e.to_legacy() for e in exits]


def compute_position_size(
    capital: float,
    risk_pct: float,
    entry_price: float,
    sl_price: float,
) -> PositionSizeResult:
    result = calc_position_size(capital, risk_pct, entry_price, sl_price)
    return PositionSizeResult.from_legacy(result)


def run_bollinger_scalping(
    df: pd.DataFrame,
) -> tuple[list[StrategySignal], list[StrategyExit], dict, pd.DataFrame]:
    entries, exits, backtest, df_s = strategy_bollinger_scalping(df)
    entry_contracts = [StrategySignal.from_legacy(e) for e in entries]
    exit_contracts = [StrategyExit.from_legacy(e) for e in exits]
    return entry_contracts, exit_contracts, backtest, df_s


def run_reversal(
    df: pd.DataFrame,
) -> tuple[list[StrategySignal], list[StrategySignal], dict, pd.DataFrame]:
    shorts, longs, backtest, df_s = strategy_reversal(df)
    short_contracts = [StrategySignal.from_legacy(s) for s in shorts]
    long_contracts = [StrategySignal.from_legacy(l) for l in longs]
    return short_contracts, long_contracts, backtest, df_s


def run_fibonacci_swing(
    df: pd.DataFrame,
) -> tuple[list[StrategySignal], list[StrategyExit], dict, pd.DataFrame, dict]:
    entries, exits, backtest, df_s, fib_data = strategy_fibonacci_swing(df)
    entry_contracts = [StrategySignal.from_legacy(e) for e in entries]
    exit_contracts = [StrategyExit.from_legacy(e) for e in exits]
    return entry_contracts, exit_contracts, backtest, df_s, fib_data

