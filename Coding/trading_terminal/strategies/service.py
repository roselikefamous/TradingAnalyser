import pandas as pd
import numpy as np

from strategies import (
    calc_position_size,
    strategy_bollinger_scalping,
    strategy_fibonacci_swing,
    strategy_reversal,
    strategy_price_action,
    strategy_candlestick_reversal,
)
from trading_terminal.contracts import (
    PositionSizeResult,
    StrategyExit,
    StrategySignal,
    WalkForwardFoldResult,
    WalkForwardSummary,
)


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


def run_price_action(
    df: pd.DataFrame,
) -> tuple[list[StrategySignal], list[StrategySignal], dict, pd.DataFrame]:
    shorts, longs, backtest, df_s = strategy_price_action(df)
    short_contracts = [StrategySignal.from_legacy(s) for s in shorts]
    long_contracts = [StrategySignal.from_legacy(l) for l in longs]
    return short_contracts, long_contracts, backtest, df_s


def run_candlestick_reversal(
    df: pd.DataFrame,
) -> tuple[list[StrategySignal], list[StrategySignal], dict, pd.DataFrame]:
    shorts, longs, backtest, df_s = strategy_candlestick_reversal(df)
    short_contracts = [StrategySignal.from_legacy(s) for s in shorts]
    long_contracts = [StrategySignal.from_legacy(l) for l in longs]
    return short_contracts, long_contracts, backtest, df_s


def run_walk_forward_backtest(
    df: pd.DataFrame,
    strategy_name: str = "bollinger_scalping",
    train_size: int = 252,
    test_size: int = 63,
    step_size: int = 63,
) -> tuple[list[WalkForwardFoldResult], WalkForwardSummary]:
    """Run rolling walk-forward validation on out-of-sample windows.

    Note: currently supports strategies with realized exits only.
    """
    if train_size <= 0 or test_size <= 0 or step_size <= 0:
        raise ValueError("train_size, test_size and step_size must be > 0")
    if len(df) < train_size + test_size:
        raise ValueError("Not enough rows for one walk-forward fold")

    strategy_map = {
        "bollinger_scalping": strategy_bollinger_scalping,
        "fibonacci_swing": strategy_fibonacci_swing,
        "reversal": strategy_reversal,
        "price_action": strategy_price_action,
        "candlestick_reversal": strategy_candlestick_reversal,
    }
    if strategy_name not in strategy_map:
        raise ValueError(
            "Unsupported strategy for walk-forward."
        )

    runner = strategy_map[strategy_name]
    fold_results: list[WalkForwardFoldResult] = []
    fold_id = 1
    train_end = train_size

    while train_end + test_size <= len(df):
        train_start = train_end - train_size
        test_start = train_end
        test_end = train_end + test_size

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]

        train_out = runner(train_df)
        test_out = runner(test_df)

        if strategy_name in {"bollinger_scalping", "fibonacci_swing"}:
            train_entries = train_out[0] if train_out else []
            test_entries = test_out[0] if test_out else []
            test_exits = test_out[1] if len(test_out) > 1 else []
            test_backtest = test_out[2] if len(test_out) > 2 else {}
        else:
            train_shorts = train_out[0] if train_out else []
            train_longs = train_out[1] if len(train_out) > 1 else []
            train_entries = train_shorts + train_longs

            test_shorts = test_out[0] if test_out else []
            test_longs = test_out[1] if len(test_out) > 1 else []
            test_entries = test_shorts + test_longs
            test_exits = _simulate_exits_for_signals(test_df, test_entries)
            test_backtest = _calc_simulated_backtest_stats(test_entries, test_exits)

        fold_results.append(
            WalkForwardFoldResult(
                fold=fold_id,
                train_start=pd.Timestamp(train_df.index[0]),
                train_end=pd.Timestamp(train_df.index[-1]),
                test_start=pd.Timestamp(test_df.index[0]),
                test_end=pd.Timestamp(test_df.index[-1]),
                train_signals=int(len(train_entries)),
                test_signals=int(len(test_entries)),
                test_exits=int(len(test_exits)),
                test_win_rate=float(test_backtest.get("win_rate", 0.0)),
                test_avg_rr=float(test_backtest.get("avg_rr", 0.0)),
                test_net_pnl_pct=float(test_backtest.get("net_pnl_pct", 0.0)),
            )
        )

        fold_id += 1
        train_end += step_size

    profitable_folds = sum(1 for f in fold_results if f.test_net_pnl_pct > 0)
    folds_n = len(fold_results)

    summary = WalkForwardSummary(
        strategy=strategy_name,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        folds=folds_n,
        profitable_folds=profitable_folds,
        pass_rate_pct=(profitable_folds / folds_n * 100) if folds_n else 0.0,
        avg_oos_net_pnl_pct=float(np.mean([f.test_net_pnl_pct for f in fold_results])) if fold_results else 0.0,
        avg_oos_win_rate=float(np.mean([f.test_win_rate for f in fold_results])) if fold_results else 0.0,
        avg_oos_rr=float(np.mean([f.test_avg_rr for f in fold_results])) if fold_results else 0.0,
    )
    return fold_results, summary


def _simulate_exits_for_signals(df: pd.DataFrame, signals: list[dict]) -> list[dict]:
    exits: list[dict] = []
    if df.empty:
        return exits

    for sig in signals:
        entry_date = pd.Timestamp(sig["Date"])
        if entry_date not in df.index:
            candidates = df.index[df.index >= entry_date]
            if len(candidates) == 0:
                continue
            start_pos = int(df.index.get_loc(candidates[0]))
        else:
            start_pos = int(df.index.get_loc(entry_date))

        side = str(sig.get("Type", "LONG")).upper()
        tp = float(sig.get("TP", sig["Entry"]))
        sl = float(sig["SL"])
        exit_payload = None

        for i in range(start_pos, len(df)):
            row = df.iloc[i]
            hit_tp = row["High"] >= tp if side == "LONG" else row["Low"] <= tp
            hit_sl = row["Low"] <= sl if side == "LONG" else row["High"] >= sl

            if hit_tp and hit_sl:
                # Conservative tie-break when both levels are touched in same bar.
                exit_payload = {"Date": df.index[i], "Price": sl, "Reason": "SL/TP same bar (conservative SL)"}
                break
            if hit_sl:
                exit_payload = {"Date": df.index[i], "Price": sl, "Reason": "SL Sim"}
                break
            if hit_tp:
                exit_payload = {"Date": df.index[i], "Price": tp, "Reason": "TP Sim"}
                break

        if exit_payload is None:
            exit_payload = {
                "Date": df.index[-1],
                "Price": float(df["Close"].iloc[-1]),
                "Reason": "EoP Mark-to-close",
            }

        exits.append(exit_payload)

    return exits


def _calc_simulated_backtest_stats(
    signals: list[dict],
    exits: list[dict],
    fee_bps: float = 5.0,
    slippage_bps: float = 3.0,
) -> dict:
    if not signals:
        return {
            "total_signals": 0,
            "total_exits": 0,
            "win_rate": 0.0,
            "avg_rr": 0.0,
            "net_pnl_pct": 0.0,
            "winners": 0,
            "losers": 0,
        }

    paired = min(len(signals), len(exits))
    winners = 0
    losers = 0
    rr_vals = []
    net_pnl_pct = 0.0

    for i in range(paired):
        sig = signals[i]
        ex = exits[i]
        entry = float(sig["Entry"])
        sl = float(sig["SL"])
        exit_price = float(ex["Price"])
        side = str(sig.get("Type", "LONG")).upper()

        risk = abs(entry - sl)
        if risk <= 0:
            continue

        pnl = (exit_price - entry) if side == "LONG" else (entry - exit_price)
        total_cost_pct = (2 * (fee_bps + slippage_bps)) / 10000
        pnl_pct = (pnl / entry) - total_cost_pct
        rr = (pnl / risk) - (total_cost_pct / (risk / entry))

        rr_vals.append(rr)
        net_pnl_pct += pnl_pct * 100
        if pnl_pct > 0:
            winners += 1
        else:
            losers += 1

    total_trades = winners + losers
    return {
        "total_signals": len(signals),
        "total_exits": len(exits),
        "win_rate": (winners / total_trades * 100) if total_trades > 0 else 0.0,
        "avg_rr": float(np.mean(rr_vals)) if rr_vals else 0.0,
        "net_pnl_pct": float(net_pnl_pct),
        "winners": winners,
        "losers": losers,
    }
